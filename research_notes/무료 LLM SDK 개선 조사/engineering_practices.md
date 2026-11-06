# Engineering practices for a resilient, well-packaged Python LLM failover SDK (`freeai`), as of November 2026

Scope note: the target is a small, dependency-light, sync-first Python 3.12 SDK on httpx that fails over across many rate-limited free endpoints. It currently has per-model/per-provider in-memory cooldowns with exponential backoff and server hints, a sliding-window RPM limiter, a UTC-day RPD counter, a `threading.Lock`, sync `chat()` and SSE `stream()`, and `httpx.MockTransport` tests. Where a source could not be fetched directly this session, that is stated next to the finding.

**Cross-cutting finding to flag first (affects several sections):** HTTPX is effectively frozen. The `httpx` 0.28.1 release on PyPI dates from 2024-12-06, and newer uploads are only `1.0.devN` pre-releases. Pydantic now maintains a fork named **`httpx2`** (2.16.0, 2026-10-21), and both `openai-python` and `anthropic-sdk-python` have moved to it ([httpx PyPI JSON](https://pypi.org/pypi/httpx/json); [httpx2 README](https://github.com/pydantic/httpx2); [openai-python pyproject](https://github.com/openai/openai-python/blob/main/pyproject.toml)). Details under the Streaming, Testing and Packaging questions.

## 1. Resilience algorithms: circuit breaker vs cooldown, backoff and jitter, retry budgets, hedging, health scoring, outlier detection. Which fit a client-side SDK?

### Takeaway
For a client SDK that fails over across many free endpoints, use a **per-endpoint cooldown that escalates like Envoy's ejection multiplier, with a single half-open probe**. Add **full-jitter capped exponential backoff** (honoring `Retry-After`), a **bounded per-call attempt budget plus a process-wide retry token bucket** (gRPC/AWS/Finagle style), and **EWMA latency/success scoring with power-of-two-choices** to rank healthy candidates. Statistical outlier detection (Envoy success-rate) and hedged requests do not fit: a single client sees too little traffic for the statistics, and every hedge burns scarce free-tier RPM/RPD quota.

### Cited Findings
**Backoff and jitter formulas (primary source: the AWS simulator code behind the AWS blog post)**
- Capped exponential: `min(cap, 2**n * base)`. Equal jitter: `v/2 + uniform(0, v/2)`. Full jitter: `uniform(0, v)`. Decorrelated jitter: `sleep = min(cap, uniform(base, sleep * 3))`, where `sleep` starts at `base` — [aws-samples/aws-arch-backoff-simulator `backoff_simulator.py`](https://github.com/aws-samples/aws-arch-backoff-simulator/blob/master/src/backoff_simulator.py)
- AWS blog conclusions: "Full Jitter uses less work, but slightly more time". "Equal Jitter does slightly more work than 'Full Jitter', and takes much longer". Decorrelated jitter "increases the maximum jitter based on the last random value". These quotes came from search-result extracts because direct fetches of aws.amazon.com timed out this session — [AWS Architecture Blog: Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

**Client-side retry budgets in shipping SDKs**
- **AWS SDK (botocore) "standard" retry mode** (primary source code):
  - `DEFAULT_MAX_ATTEMPTS = 3`, with `_BASE = 2` and `_MAX_BACKOFF = 20`.
  - Delay is `rand(0,1) * min(2**attempt, MAX_BACKOFF)`, i.e. full jitter.
  - Retry quota is a token bucket with `INITIAL_CAPACITY = 500`. Costs: `_RETRY_COST = 5`, `_RETRY_COST_V2 = 14` (non-throttling errors), `_THROTTLING_RETRY_COST = 5`. On success the bucket gets back `_NO_RETRY_INCREMENT = 1`.
  - Sources: [botocore/retries/standard.py](https://github.com/boto/botocore/blob/develop/botocore/retries/standard.py); [botocore/retries/quota.py](https://github.com/boto/botocore/blob/develop/botocore/retries/quota.py)
- AWS docs (via search extract; direct fetch timed out): the standard strategy makes at most 3 attempts. The bucket holds 500 tokens. A retry costs 14 tokens after a non-throttling error and 5 after a throttling error. The backoff cap is 20 s. When tokens run out, the error is returned without retrying — [AWS SDK for Python v4: config retries](https://docs.aws.amazon.com/sdk-for-python/v4/guide/config-retries.html)
- **AWS "adaptive" mode** adds a `ClientRateLimiter`. It feeds a `throttling.CubicCalculator` (a CUBIC-style rate adjuster) from a `ThrottlingErrorDetector`, so the client self-throttles its send rate after throttling responses — [botocore/retries/adaptive.py](https://github.com/boto/botocore/blob/develop/botocore/retries/adaptive.py)
- **gRPC (A6)**:
  - Retry backoff is `min(initialBackoff*backoffMultiplier**(n-1), maxBackoff) * random(0.8, 1.2)`, and `maxAttempts` values above 5 are treated as 5.
  - Retry throttling keeps a per-server `token_count` that starts at `maxTokens` (example 10). Each failure subtracts 1 and each success adds `tokenRatio` (example 0.1). Retries are disabled while `token_count <= maxTokens/2`.
  - The server-pushback header `grpc-retry-pushback-ms` gives an exact delay. A negative or unparseable value means "do not retry".
  - Source: [gRPC proposal A6: client retries](https://github.com/grpc/proposal/blob/master/A6-client-retries.md)
- **Finagle**:
  - Retries come out of a `RetryBudget` that by default allows about 20% of total requests to be retried immediately.
  - The factory is a leaky token bucket, `RetryBudget(ttl=10.seconds, minRetriesPerSec=5, percentCanRetry=0.1)`.
  - Finagle recommends using `Backoff` with the budget.
  - Source: [Finagle Clients guide](https://twitter.github.io/finagle/guide/Clients.html)
- **Google SRE book**:
  - Per-request retry budget of "up to three attempts". Per-client retry budget: a request is retried only while retries are "below 10%" of requests. Without the budget, retries can push load to about 3x; with it, about 1.1x.
  - Retry only "at the layer immediately above the layer that is rejecting them". If the whole datacenter is overloaded, don't retry, and let the error bubble up.
  - Client-side adaptive throttling tracks `requests` vs `accepts` over the last 2 minutes and starts rejecting locally once `requests` exceeds K×`accepts`. They "generally prefer the 2x multiplier".
  - Source: [Google SRE book, Handling Overload](https://sre.google/sre-book/handling-overload/)
- **Microsoft Azure guidance**:
  - Use exponential backoff with jitter for background work, and immediate or regular-interval retries for interactive work.
  - Avoid duplicated layers of retry code, and never retry endlessly.
  - "Per-request retry limits alone can't prevent a scenario where many concurrent requests each retry a few times and collectively overwhelm a struggling downstream service. A retry budget limits the aggregate retry load".
  - Source: [Azure: Transient fault handling](https://learn.microsoft.com/en-us/azure/architecture/best-practices/transient-faults)

**Circuit breaker vs cooldown**
- Azure's circuit-breaker pattern:
  - **Closed** keeps a failure counter that is time-based and "automatically resets at periodic intervals".
  - **Open** fails fast until a timeout timer expires.
  - **Half-Open** lets a limited number of requests through. After a specified number of consecutive successes the breaker closes; any failure re-opens it.
  - The guidance suggests an increasing open timeout (a few seconds at first, then longer).
  - Source: [Azure: Circuit Breaker pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)

**Outlier detection (Envoy) and load balancing (Finagle)**
- Envoy detection types:
  - Consecutive 5xx, consecutive gateway failure (502/503/504), and consecutive local-origin failure (timeouts, resets).
  - Success rate, judged against cluster mean/stdev, with `success_rate_minimum_hosts` and `success_rate_request_volume` gates.
  - Failure percentage against a flat threshold.
  - Source: [Envoy outlier detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier)
- Envoy ejection:
  - Ejection time is `base_ejection_time` × "the number of times the host has been ejected in a row", capped at `max_ejection_time`. The multiplier decreases each healthy `interval`.
  - Hosts return automatically after ejection, and active health checking is optional.
  - `max_ejection_percent` caps simultaneous ejections, and in a panic scenario ejected hosts are used anyway.
  - The arch page gives no numeric defaults; they live in the API reference.
  - Source: [Envoy outlier detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier)
- Finagle balancers:
  - The default balancer is P2C (power of two choices) + least loaded.
  - "P2C + Peak EWMA" uses a peak-sensitive moving average of RTT weighted by outstanding requests, designed "to react to slow endpoints more quickly". The page notes this assumption "breaks down in the presence of long polling clients".
  - Source: [Finagle Clients guide](https://twitter.github.io/finagle/guide/Clients.html)

**Hedging**
- gRPC hedging sends up to `maxAttempts` copies, each spaced `hedgingDelay` apart. A non-fatal status triggers the next hedge immediately, and a fatal status cancels all outstanding RPCs. Hedges after the first are sent only while the throttle `token_count` is above the threshold — [gRPC A6](https://github.com/grpc/proposal/blob/master/A6-client-retries.md)
- OpenAI states that "unsuccessful requests contribute to your per-minute limit, so continuously resending a request won't work" — [OpenAI rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits)

**Prior art: LLM-specific routers and SDKs**
- **LiteLLM Router**:
  - Cooldowns apply per deployment, not per model group, with default `cooldown_time` 5 s.
  - Without `allowed_fails`, a deployment cools down immediately on a 429, at >50% failures in the current minute, or on 401/404/408.
  - With `allowed_fails`, it cools down after more than N failures within `cooldown_time`.
  - Routing strategies: `simple-shuffle` (weighted pick, the default and recommended), rate-limit-aware v2, latency-based, least-busy, and lowest-cost.
  - Fallbacks are ordered (`order=1` → `order=2`…), with `context_window_fallbacks`. `enable_weighted_failover` (async-only) excludes the failed deployment and re-picks from peers, capped by `max_fallbacks` (default 5).
  - A request that finds all `max_parallel_requests` slots full fails immediately with a 429; there is no queue.
  - Redis is used in production to share cooldowns and TPM/RPM.
  - Source: [LiteLLM routing docs](https://docs.litellm.ai/docs/routing)
- **openai-python** (primary source):
  - Constants: `DEFAULT_MAX_RETRIES = 2`, `INITIAL_RETRY_DELAY = 0.5`, `MAX_RETRY_DELAY = 8.0`, `MAX_RETRY_AFTER_DELAY = 120`.
  - Server hints: it reads `retry-after-ms`, then `retry-after` (seconds or an HTTP date), and honors values with 0 < delay ≤ 120 s. If `Retry-After` exceeds 120 s it does not retry at all. A non-standard `x-should-retry: true/false` header overrides everything else.
  - Otherwise it retries 408, 409, 429 and ≥500. Jitter multiplies the delay by `1 - 0.25*random()`.
  - Sources: [openai `_base_client.py`](https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py); [openai `_constants.py`](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py)
- The OpenAI README says requests are "retried only when their body can be safely resent", and "Stream consumption is not automatically retried, because replaying a request could duplicate output already delivered to your application" — [openai-python README](https://github.com/openai/openai-python/blob/main/README.md)
- anthropic-sdk-python has the same constants (`DEFAULT_MAX_RETRIES = 2`, `INITIAL_RETRY_DELAY = 0.5`, `MAX_RETRY_DELAY = 8.0`) and also honors `retry-after-ms` and `x-should-retry` — [anthropic `_constants.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_constants.py); [anthropic `_base_client.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_base_client.py)

### Inferences
**Cooldown vs full circuit breaker**
- The existing per-model/per-provider cooldown already acts as a circuit breaker's Open state. The cheap upgrades are:
  - (a) An Envoy-style escalation, `cooldown = min(max_cd, base_cd * consecutive_ejections)`, plus decay on success instead of pure doubling. The doubling the SDK already does is fine; add a cap and a decay.
  - (b) A **half-open single-flight probe**: when a cooldown expires, mark the endpoint `probing` so only one in-flight call tests it. On success, reset; on failure, re-eject with a higher multiplier.
  - (c) **Error classification** decides which cooldown applies:
    - 429 with a hint → use the hint exactly.
    - 429 without a hint → escalating cooldown.
    - Daily-quota exhaustion → cooldown until that provider's reset instant.
    - 401/403 → long or permanent disable plus a warning.
    - 404 model → disable that model.
    - 5xx or timeout → short escalating cooldown.
    - 400 or a context-length error → no cooldown; fail over only to models with larger context, or raise.
  - A separate three-state breaker class adds little beyond this.

**Backoff**
- Use full jitter (`uniform(0, min(cap, base*2**n))`), the AWS-recommended low-work option, for retries against the *same* endpoint.
- In a failover SDK, most "retries" should be **moves to the next candidate** with zero delay. Sleeping only makes sense when every candidate is cooling down; then sleep until the earliest cooldown expiry, bounded by the caller's deadline.
- Server hints (`retry-after-ms`, `retry-after`, `x-ratelimit-reset-*`) should override computed backoff. Like openai-python, treat a very large Retry-After (for example over 120 s, or a daily reset) as "skip this endpoint" rather than sleeping.

**Retry budget**
- Two layers are proportionate here:
  - (1) **Per-call**: `max_attempts` (for example the number of candidates, capped at about 5–8) plus a total `deadline`.
  - (2) **Per-process**: a gRPC-style token bucket (`max_tokens=10, ratio=0.1`) or a Finagle-style "retries ≤ 10–20% of calls". Once the budget is exhausted, raise `AllProvidersFailed` immediately instead of sweeping every endpoint.
- This prevents a fleet of scripts from hammering many free endpoints when one shared upstream (for example OpenRouter) is down.
- Retry at one layer only (SRE and Azure): if the SDK fails over, it should not also let httpx or a provider SDK retry underneath.

**Hedging**
- Hedging fits poorly with free tiers. Each hedge consumes RPM and RPD on another provider, and failed requests still count against quotas (OpenAI states this explicitly).
- If offered at all: opt-in only, at most 2 attempts, `hedging_delay` ≥ p95 time-to-first-token, gated by the retry token bucket (as in gRPC). Cancel the loser on the first chunk.

**Health scoring**
- Keep a per-endpoint EWMA of latency (or TTFT for streams) and of success (1/0), with α ≈ 0.2–0.3 or time-decayed.
- Rank by a score such as `success_ewma / latency_ewma`, adjusted for quota headroom. Pick with P2C (sample two eligible candidates and take the better) so load spreads without herding.
- Envoy success-rate ejection needs minimum host counts and request volumes per interval, which a single SDK process rarely reaches. Use consecutive-failure ejection instead.
- Peak-EWMA's caveat about long-polling suggests using **TTFT rather than total stream duration** as the latency signal.

### Gaps
- I could not fetch the AWS blog and the AWS SDK docs pages directly (timeouts and DNS failures). The formulas are verified from AWS's own simulator source and from botocore source; the prose conclusions come from search extracts.
- Envoy's numeric defaults (for example `consecutive_5xx`, `base_ejection_time`, `max_ejection_percent`) are not on the arch page. They are in the API reference, which I did not fetch.
- Finagle's failure-accrual policy parameters (consecutive-failure count, dead-time backoff) were not in the fetched Clients page.
- I found no primary source that benchmarks EWMA α values or hedging specifically for LLM APIs.

## 2. Rate limiting: token bucket vs GCRA vs sliding-window log for RPM and TPM; estimating tokens before a request; learning limits from `x-ratelimit-*` headers (adaptive/AIMD)

### Takeaway
Use **GCRA or a token bucket (constant memory, smooth) for RPM and TPM**, which matches how providers such as Anthropic actually enforce limits. Use a **fixed-window counter keyed to each provider's documented reset boundary for RPD**: Gemini resets at **midnight Pacific**, not UTC. Estimate tokens with a dependency-free **chars/4 (plus `max_tokens`) heuristic** and reconcile with actual `usage` after the response. Treat `x-ratelimit-*` headers as **authoritative corrections** of local state, parsed per provider, because their semantics differ (Groq's `limit-requests` is RPD, for example). An adaptive CUBIC/AIMD loop is optional.

### Cited Findings
**How providers enforce limits**
- Anthropic:
  - "The API uses the token bucket algorithm… capacity is continuously replenished up to your maximum limit, rather than being reset at fixed intervals". A 60 RPM limit "might be enforced as 1 request per second", so short bursts can trigger 429s.
  - Limits are RPM, ITPM and OTPM. "ITPM rate limits are estimated at the beginning of each request, and the estimate is adjusted during the request". OTPM counts actual generated tokens, and "max_tokens does not factor into OTPM".
  - Cached reads do not count toward ITPM.
  - Headers: `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` plus `retry-after`.
  - Some 429s for exhausted access carry no `retry-after`, and retrying "fails until access resumes". There are also "acceleration limits" on sharp usage increases.
  - Source: [Anthropic rate limits](https://docs.claude.com/en/api/rate-limits)
- OpenAI:
  - Headers are `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`, plus `-project-tokens` variants.
  - "Your rate limit is calculated as the maximum of max_tokens and the estimated number of tokens based on the character count of your request".
  - Unsuccessful requests count toward per-minute limits.
  - Some model families share one limit.
  - Source: [OpenAI rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits)
- Groq:
  - Limits are RPM, RPD, TPM, TPD, ASH and ASD; some organizations also have separate ITPM/OTPM.
  - `x-ratelimit-limit-requests`, `-remaining-requests` and `-reset-requests` "always refer to Requests Per Day (RPD)". The `-tokens` headers always refer to TPM.
  - Reset values use Go-duration strings such as `2m59.56s` and `7.66s`.
  - `retry-after` (seconds) is "only set if you hit the rate limit and status code 429 is returned".
  - Source: [Groq rate limits](https://console.groq.com/docs/rate-limits)
- Gemini API: daily quotas "reset at midnight Pacific time" — [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

**Emerging header standard**
- IETF `draft-ietf-httpapi-ratelimit-headers-11` defines `RateLimit-Policy` (for example `"permin";q=50;w=60,"perhr";q=1000;w=3600`, with an optional `qu="content-bytes"` unit and a `pk` partition key) and `RateLimit` (for example `"default";r=50;t=30`, where `r` is remaining and `t` is seconds until reset).
  - Status: last updated 2026-05-23, expires 2026-11-24, and still a WG Document with IESG state "I-D Exists".
  - An HTTPDIR early review of -10 said "Not ready".
  - Source: [IETF datatracker: RateLimit header fields](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)

**Algorithms in Python libraries**
- PyrateLimiter 4.5.1:
  - `SlidingWindowLog` is the default. It is exact but "stores one entry per consumed unit".
  - `FixedWindow` lets "up to 2 * limit" pass across a boundary. The docs recommend it "when you are mirroring an upstream API that genuinely resets on a boundary — a quota that refreshes on the hour, or at midnight UTC".
  - `GCRA`/`TokenBucket` needs "a couple of numbers, whatever the traffic", with constant memory and a smooth output rate.
  - Items can carry a `weight`, and weight W consumes W slots atomically, which is how TPM can be modeled.
  - Source: [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)
- PyrateLimiter notes that a Redis sorted-set sliding log "can grow large" for daily or monthly quotas — [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)
- `limits` 5.8.0 offers fixed window (the boundary burst is documented), moving window and sliding-window counter, with `hit`/`test`/`get_window_stats` methods and identical sync and async APIs — [limits on PyPI](https://pypi.org/project/limits/)
- `aiolimiter` 1.3.0 is an "asyncio rate limiter, a leaky bucket implementation" — [aiolimiter on PyPI](https://pypi.org/project/aiolimiter/)
- GCRA is a leaky-bucket variant from ATM networks. It tracks a "theoretical arrival time" (TAT) and needs no background "drip" process, which brandur identifies as the naive leaky bucket's main weakness — [brandur.org: Rate Limiting, Cells, and GCRA](https://brandur.org/rate-limiting)
- Adaptive client rate: botocore's adaptive mode adjusts the client send rate with a CUBIC calculator driven by throttling detection — [botocore adaptive.py](https://github.com/boto/botocore/blob/develop/botocore/retries/adaptive.py)

**Token estimation**
- tiktoken is a fast BPE tokenizer (for example `o200k_base`, `encoding_for_model("gpt-4o")`), "between 3-6x faster than a comparable open source tokeniser" — [tiktoken README](https://github.com/openai/tiktoken)

### Inferences
**Recommended limiter design (stdlib only)**
- One GCRA cell per (provider, metric) for RPM and TPM: state is a single float `tat` per key. Admission check: `allow if now >= tat - burst_tolerance`, then `tat = max(tat, now) + cost * emission_interval`.
- GCRA naturally supports a weighted cost (tokens) and gives an exact `retry_at` for scheduling the next attempt. It replaces the sliding-window log (O(n) memory per window) and serializes trivially into SQLite or Redis (one row per key).
- RPD should remain a fixed-window counter keyed by `(provider, reset_period_id)`. The reset timezone/boundary must be **per-provider config**: a UTC-day counter is wrong for Gemini (Pacific midnight), and Groq reports RPD reset as a duration.

**TPM accounting**
- Reserve `estimate = ceil(len(prompt_chars)/4) + max_tokens` (or a default output cap) before sending, then correct with actual `usage.prompt_tokens + completion_tokens`. This mirrors OpenAI's own "max of max_tokens and character-based estimate".
- For providers with separate ITPM/OTPM (Anthropic, some Groq orgs), track them separately and charge output after the fact.
- Offer `tiktoken` only as an optional extra (`freeai[tokens]`), because it is a compiled dependency. Most free endpoints serve non-OpenAI models (Llama, Qwen, Gemma), where tiktoken's count is itself only an estimate.

**Header learning**
- Parse per-provider header dialects into a normalized `RateLimitSnapshot(limit, remaining, reset_at, scope)` and **overwrite** local estimates. When `remaining == 0`, set a cooldown until `reset_at`. When `limit` differs from configuration, adopt the server's value, since free-tier limits change often.
- Support the IETF `RateLimit`/`RateLimit-Policy` format opportunistically. It is still a draft, so treat it as best-effort.
- An AIMD loop (additive increase on success, multiplicative decrease on 429) is only worth it for endpoints that send **no** limit headers. Even then, a 429-triggered cooldown plus a GCRA rate set to the documented limit covers most cases.

### Gaps
- The OpenAI help-center article with the "~4 characters per token" rule of thumb could not be fetched (blocked). The rule is widely cited but not verified this session; OpenAI's rate-limit guide confirms only "estimated… based on the character count".
- I found no primary source comparing chars/4 accuracy for non-English text (for example Korean) or for Llama/Qwen tokenizers.
- **Provider-side tokenizers and token-counting endpoints were not researched.** No provider's count-tokens API or tokenizer download was fetched. Trade-off (inference, unsourced): a remote count-tokens call is itself a request that may count against RPM or RPD, the scarcest resource on free tiers. Shipping per-model local tokenizers would bloat dependencies. So the chars/4 estimate plus reconciliation from actual usage looks like the proportionate default.
- I fetched no primary source for an AIMD rate limiter in an LLM SDK. Netflix `concurrency-limits` and similar sources were not consulted.
- I did not fetch provider header formats for other free providers (Cerebras, Mistral, OpenRouter, GitHub Models); other researchers may cover them.

## 3. Sync + async parity in Python: how major SDKs do it, least duplication for a small library, asyncio concurrency safety, and stream cancellation

### Takeaway
The big SDKs (OpenAI, Anthropic) are **code-generated** (Stainless) and simply ship two parallel class hierarchies: `SyncAPIClient`/`AsyncAPIClient` and `Stream`/`AsyncStream`. That is cheap for a generator and expensive by hand. For a small hand-written library, the least duplication comes from a **sans-I/O core** (selection, cooldown, limiter math, SSE parsing and response normalization as pure functions or objects with an injected clock) plus **two thin I/O shells**. Optionally, generate the sync shell from the async one with an **unasync-style regex script**, as httpcore does.

### Cited Findings
- openai-python is "generated from our OpenAPI specification" and "offers both synchronous and asynchronous clients powered by HTTPX2" — [openai-python README](https://github.com/openai/openai-python/blob/main/README.md)
- In openai-python, `class BaseClient(Generic[_HttpxClientT, _DefaultStreamT])` holds the shared logic: retry calculation, header parsing and `_should_retry`. `class SyncAPIClient(BaseClient[httpx2.Client, Stream[Any]])` (line ~1001) and `class AsyncAPIClient(BaseClient[httpx2.AsyncClient, AsyncStream[Any]])` (line ~1622) duplicate the I/O paths in a 2,391-line file — [openai `_base_client.py`](https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py)
- anthropic-sdk-python has the same structure (`SyncAPIClient` at line ~1071, `AsyncAPIClient` at line ~1851) — [anthropic `_base_client.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_base_client.py)
- openai-python `_streaming.py` defines separate `Stream` and `AsyncStream` classes, plus a shared, I/O-free `SSEDecoder`/`_SSELineDecoder` used by both — [openai `_streaming.py`](https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py)
- httpcore keeps a `scripts/unasync.py` that generates sync code from async code with regex substitutions, for example:
  - `'async def'→'def'`, `'await '→''`, `'async with'→'with'`, `'async for'→'for'`
  - `'aclose'→'close'`, `'__aenter__'→'__enter__'`, `'__aiter__'→'__iter__'`, `'AsyncIterator'→'Iterator'`
  - `'Async([A-Z]…)'→'\2'`, `'AutoBackend'→'SyncBackend'`
  - test markers such as `@pytest.mark.anyio` are removed
  - Source: [httpcore `scripts/unasync.py`](https://github.com/encode/httpcore/blob/master/scripts/unasync.py)
- The sans-I/O approach means "network protocol implementations written in Python that perform no I/O (this means libraries that operate directly on text or bytes…)" — [Sans I/O documentation](https://sans-io.readthedocs.io/)
- asyncio synchronization primitives "are not thread-safe, therefore they should not be used for OS thread synchronization (use threading for that)". `asyncio.Lock` is "Not thread-safe" — [Python docs: asyncio synchronization primitives](https://docs.python.org/3/library/asyncio-sync.html)
- `contextlib.aclosing()` "supports deterministic cleanup of async generators when they happen to exit early by break or an exception" — [Python docs: contextlib.aclosing](https://docs.python.org/3/library/contextlib.html#contextlib.aclosing)
- openai's `AsyncStream` exposes `close()`/`aclose()` and `__aexit__`, and both close the underlying response (`await self.response.aclose()`). `Stream.close()` and `__exit__` do the same for sync — [openai `_streaming.py`](https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py)
- PyrateLimiter's `Limiter` is "thread-safe via `RLock` (and `asyncio.Lock` when async)". "The same API works in both; async paths never block the event loop". It provides `BucketAsyncWrapper` to "make any sync bucket async-safe" — [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)
- `limits` "provides identical APIs for use in sync and async codebases" — [limits on PyPI](https://pypi.org/project/limits/)

### Inferences
**Recommended structure for `freeai`**
- `freeai/_core/`, all pure with no httpx import:
  - `Router.plan(request, now) -> list[Candidate]` and `Router.on_result(candidate, outcome, headers, now)`
  - `Limiter.try_acquire(key, cost, now) -> Wait|Ok`
  - `SSEParser.feed(bytes) -> list[Event]`
  - `normalize(provider, json) -> ChatResponse`
- `freeai/_sync.py` and `freeai/_async.py` contain only "loop over the plan → send → feed the core → sleep/raise", using `httpx.Client` vs `httpx.AsyncClient` and `time.sleep` vs `asyncio.sleep`.
- Each shell is likely about 150–300 lines. That is small enough to hand-maintain twice, or to generate with a roughly 30-line unasync script plus a CI check that the generated file is up to date.
- This is less machinery than a Stainless-style generator and keeps most logic testable without any transport.

**Locks**
- Keep **one `threading.Lock`** guarding the core state, held only for non-blocking microsecond critical sections with **no `await` and no I/O inside**. That is safe from both threads and coroutines.
- `asyncio.Lock` is not thread-safe, so it would break when a sync client and an async client share one state store in a multithreaded app.
- Never sleep while holding the lock. Compute `retry_at` under the lock, release it, then `await asyncio.sleep(...)` or `time.sleep(...)`.

**Cancellation**
- In the async stream, wrap the body in `try/finally: await response.aclose()`. The finally block must run on `asyncio.CancelledError` and on early `break`.
- Do not swallow `CancelledError`. It must re-raise after cleanup, and it must not be counted as a provider failure: no cooldown, and refund any reserved TPM estimate.
- Document `async with aclosing(client.astream(...))`, or return an object with `__aenter__`/`__aexit__` (as openai does), so early consumer exit closes the connection deterministically rather than at garbage collection.

### Gaps
- I did not fetch the `python-trio/unasync` package docs. The httpcore script is the concrete precedent verified here.
- I did not fetch Stainless's own documentation describing its generation approach. "Generated" is inferred from openai's README ("generated from our OpenAPI specification") and `X-Stainless-*` constants.
- I found no primary benchmark on the event-loop impact of a short `threading.Lock` in asyncio code. The safety claim is reasoning from the asyncio docs, not a measured source.

## 4. Shared/persistent state across processes: SQLite (WAL) vs file locks vs Redis; pluggable state-store interfaces; examples from rate-limit libraries

### Takeaway
Define a small **`StateStore` protocol** and ship three tiers:
- **in-memory** (default);
- **SQLite** (stdlib `sqlite3`, WAL, `BEGIN IMMEDIATE`), as zero-dependency cross-process state on **one host**;
- **Redis** as an optional extra for multi-host.

Store **wall-clock epoch timestamps** in shared state, because monotonic clocks are per process. The quota counter (RPD/TPM), not only cooldowns, is the real reason to share state: free-tier daily quotas are per account, across all processes.

### Cited Findings
- SQLite WAL: "readers do not block writers and a writer does not block readers". "All processes using a database must be on the same host computer; WAL does not work over a network filesystem". WAL mode can still return `SQLITE_BUSY` in some cases, and it requires checkpointing — [SQLite: Write-Ahead Logging](https://www.sqlite.org/wal.html)
- PyrateLimiter backend matrix:

  | Backend | Persistent | Multi-process | Async | Notes |
  |---|---|---|---|---|
  | `InMemoryBucket` | No | No | (wrap) | |
  | `SQLiteBucket` | Yes | Yes, via `filelock` | No | "persistence / one host, many processes" |
  | `RedisBucket` | Yes | Yes | Yes (sync and async) | distributed |
  | `PostgresBucket` | Yes | Yes | No | |
  | `MultiprocessBucket` | No | single `multiprocessing` pool | (wrap) | |

  `SQLiteBucket.init_from_file(..., use_file_lock=True)` uses `filelock` "to share one DB file across processes". The extras are `[redis]`, `[postgres]`, `[filelock]` and `[all]` — [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)
- PyrateLimiter on clocks: "GCRA measures elapsed time, so state shared between machines needs a shared clock. `RedisStateStore` defaults the bucket to `WallClock` (epoch ms)… local stores default to `MonotonicClock`". A `buffer_ms` slack (default 50) absorbs clock drift — [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)
- `limits` supports storage backends "Redis, Memcached & MongoDB" (plus memory) behind one strategy API — [limits on PyPI](https://pypi.org/project/limits/)
- LiteLLM: "In production, litellm supports using Redis as a way to track cooldown server and usage (managing tpm/rpm limits)" — [LiteLLM routing docs](https://docs.litellm.ai/docs/routing)

### Inferences
**Protocol sketch (sync; the async variant is mirrored or wrapped)**

```python
class StateStore(Protocol):
    def acquire(self, key: str, cost: float, rate: Rate, now: float) -> float: ...  # returns 0 or seconds-to-wait (GCRA, atomic)
    def incr_window(self, key: str, window_id: str, cost: int, limit: int) -> bool: ...  # RPD fixed window
    def set_cooldown(self, key: str, until: float, reason: str) -> None: ...
    def cooldowns(self, keys: Iterable[str], now: float) -> dict[str, float]: ...
    def record_health(self, key: str, ok: bool, latency: float, now: float) -> None: ...  # EWMA; may stay process-local
```

**Implementation tiers**
- **Memory**: dicts plus `threading.Lock` (the current design).
- **SQLite**:
  - Use `PRAGMA journal_mode=WAL`, `busy_timeout` (for example 5000 ms), and short `BEGIN IMMEDIATE` transactions per acquire. The GCRA row needs a single `UPDATE … SET tat=? WHERE key=?`, which avoids needing `filelock`.
  - Default the path to a user cache dir (for example `~/.cache/freeai/state.db`). Document "same host, local disk only" because of the WAL constraint.
  - Expire rows lazily.
- **Redis** (`freeai[redis]`): one Lua script per operation (GCRA, window incr with `EXPIRE`, cooldown `SET … PXAT`) for atomicity.

**What to share vs keep local**
- Share cooldowns and quota counters, since they are account-scoped facts.
- Health EWMAs can stay process-local (cheap, noisy, and not account-scoped).
- **File locks alone** (for example JSON plus `filelock`) are simpler but serialize every request and are easy to corrupt on crash. SQLite gives atomicity for free with the stdlib.

### Gaps
- I found no published benchmark of SQLite WAL throughput under many-process contention for rate-limit workloads in this research.
- I did not fetch the `limits` storage docs for the exact algorithm-to-storage support matrix (for example which strategies support MongoDB or etcd), or the `filelock` docs.
- I did not verify whether Python 3.12's `sqlite3` default isolation and autocommit behavior affects `BEGIN IMMEDIATE` usage. That needs the stdlib docs.

## 5. Streaming robustness: SSE parsing edge cases, idle/read timeouts vs total deadline, closing connections on early exit, httpx timeout configuration

### Takeaway
Implement (or adopt) a **spec-exact incremental SSE parser**:
- CR/LF/CRLF line endings, including a CR split across chunks;
- `:` comment/keep-alive lines ignored;
- multi-line `data` joined with `\n`;
- `event`/`id`/`retry` fields, and no dispatch on an empty data buffer;
- a per-event size cap.

httpx (and httpx2) only offer **connect/read/write/pool** timeouts, all of them inactivity timeouts with no total deadline. The SDK must therefore enforce a **total deadline** itself and should use the read timeout as the stream idle timeout. Fail over **only before the first content chunk** has been yielded, and always close the response in `finally`.

### Cited Findings
**SSE spec (WHATWG)**
- Grammar: `stream = [ bom ] *event`, and `end-of-line = ( cr lf / cr / lf )`.
- Parsing: a blank line dispatches the event. A line starting with `:` is ignored (comment). Otherwise, split at the first colon and strip one leading space from the value. A line with no colon is a field with an empty value.
- `event` sets the type. `data` appends the value plus LF. `id` is set only if the value has no NULL. `retry` is set only if the value is all ASCII digits. Unknown fields are ignored.
- On dispatch the trailing LF is removed; an empty data buffer means nothing is dispatched.
- Servers can send "a comment line… every 15 seconds or so" to defeat proxy timeouts.
- Source: [WHATWG HTML: Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)

**Reference implementations**
- openai-python's `_SSELineDecoder` "incrementally split[s] CR, LF, and CRLF without copying growing prefixes". It tracks `_skip_lf` so a CR at the end of one chunk followed by LF at the start of the next counts as one line ending.
- Its `SSEDecoder` handles `event`/`data`/`id`/`retry` and `:` comments, and the stream stops on `data: [DONE]`.
- Source: [openai `_streaming.py`](https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py)
- **httpx2 native SSE** (added in 2.5.0, 2026-06-25):
  - `client.sse(url, method="POST", json=...)` is a context manager yielding an `EventSource`. It sets `Accept: text/event-stream` and `Cache-Control: no-store` automatically.
  - Each `ServerSentEvent` exposes `event` (default `"message"`), `data` (multiple `data:` lines joined with `\n`), `id` (persists across events) and `retry`, plus `.json()`.
  - A non-`text/event-stream` content type raises `SSEError`, a subclass of `TransportError`.
  - `max_event_size` defaults to 1 MiB per event (`None` disables it). It was added in 2.10.0.
  - Sources: [httpx2 docs: sse.md](https://github.com/pydantic/httpx2/blob/main/docs/sse.md); [httpx2 CHANGELOG](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)
- For classic httpx, the third-party `httpx-sse` package (0.4.3) is the equivalent — [httpx-sse on PyPI](https://pypi.org/project/httpx-sse/)

**Timeouts**
- httpx2 (identical to httpx 0.28.1) "enforce[s] timeouts everywhere by default… after 5 seconds of network inactivity". There are four timeout types:
  - `connect`;
  - `read`, the maximum wait "for a chunk of data to be received";
  - `write`;
  - `pool`, the wait to acquire a connection from the pool, related to `limits`.
  - Example: `Timeout(10.0, connect=60.0)`.
  - Source: [httpx2 docs: timeouts](https://github.com/pydantic/httpx2/blob/main/docs/advanced/timeouts.md)
- openai-python's default is `Timeout(timeout=600, connect=5.0)` (10 minutes), with the README example `Timeout(60.0, read=5.0, write=10.0, connect=2.0)`. During stream consumption, read timeouts raise `APITimeoutError` and other failures raise `APIConnectionError`, with the original error in `__cause__`. "Stream consumption is not automatically retried, because replaying a request could duplicate output" — [openai `_constants.py`](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py); [openai-python README](https://github.com/openai/openai-python/blob/main/README.md)
- httpx2 2.12.0: "Bound peak memory while streaming compressed responses and close response streams when decoding fails" — [httpx2 CHANGELOG](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)

**Defining the first chunk**
- The OTel GenAI conventions define a "chunk" for time-to-first-chunk as "a decoded provider stream event carrying new, non-empty output". Instrumentation "MUST NOT treat empty start events or events carrying only usage, roles, IDs, signatures, keep-alives, progress/status updates, or end markers as chunks" — [OTel GenAI client inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)

### Inferences
**Parser edge cases to unit-test**
- CRLF split across chunks, and a lone CR line ending.
- A leading BOM.
- `data:` with no space, and `data` with no colon (an empty value).
- Multiple `data:` lines.
- `:` keep-alive comments. OpenRouter is known to send `: OPENROUTER PROCESSING` comment lines, but that comes from prior knowledge and was not fetched this session.
- `event: error` frames carrying a JSON error mid-stream.
- A final event without a trailing blank line (spec: discard; some servers rely on flush, so decide and document).
- `[DONE]`.
- An oversized event (cap it, as httpx2 does).
- Non-UTF-8 bytes (decode with `errors="replace"`).
- HTTP 200 with a JSON error body and a non-SSE content type, so the content type must be checked before parsing.

**Timeout policy**
- `connect=5s` (fail over fast on dead hosts).
- `write=10s`.
- `pool=5s`.
- `read` = idle timeout: about 30–60 s for streams, but lower for TTFT if desired. Free endpoints with long queues can stall before the first token, so a separate `first_token_timeout` is valuable.
- Total `deadline` enforced by the SDK:
  - Sync: compute `remaining = deadline - monotonic()` before each attempt and pass `timeout=min(configured, remaining)` per request. While iterating, check the deadline per chunk.
  - Async: wrap each attempt in `asyncio.timeout(remaining)`.

**Failover semantics**
- Before the first content chunk is yielded to the caller, any error (including an `event: error` frame or an idle timeout) can transparently fail over to the next candidate.
- After the first chunk, raise `StreamInterrupted(partial_text, provider, model)` rather than silently re-running, consistent with OpenAI's rationale. An opt-in "resume by re-prompting with partial output" could come later.

**Cleanup**
- The generator body must use `with client.stream(...) as r:` or `try/finally: r.close()`, so that a `break` in user code, `GeneratorExit`, or an exception closes the socket and returns the connection to the pool.
- Expose the stream as a context-manager object (like openai's `Stream`) so users can close it deterministically.

**Using httpx2's `client.sse()`**
- If the SDK migrates to httpx2, it can delete its own SSE parser. Keeping a tiny local parser is still reasonable for byte-level control and for staying compatible with httpx.

### Gaps
- I did not verify whether httpx2's parser dispatches a final unterminated event at EOF, or how it handles a BOM; the docs page doesn't say.
- I found no source documenting the keep-alive or comment behavior of specific free providers' SSE streams (beyond the general WHATWG advice). Other researchers covering providers may have this.

## 6. Observability: Python logging conventions for libraries, OpenTelemetry GenAI semantic conventions (status as of 2026), hooks/callbacks, exposing usage and cost

### Takeaway
Log through `logging.getLogger("freeai")` and its children with only a `NullHandler` attached, at DEBUG for per-attempt detail and WARNING for disabled keys or exhausted quotas, and **never log secrets**. The OTel GenAI conventions **moved out of the core semconv repo** into `open-telemetry/semantic-conventions-genai` and remain **"Development"** status as of November 2026; names have churned (`gen_ai.client.inference.*`). Ship a **dependency-free hook/callback API** as the stable contract, with an optional `freeai[otel]` extra that maps hook events onto the GenAI conventions.

### Cited Findings
**Python logging for libraries**
- The HOWTO's "Configuring Logging for a Library" section recommends adding a `NullHandler` to the library's top-level logger, for example `logging.getLogger('foo').addHandler(logging.NullHandler())`. "It is strongly advised that you do not log to the root logger", and "strongly advised that you do not add any handlers other than NullHandler" to library loggers. Libraries should document their logger names — [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library)
- httpx2 renamed its logger to `httpx2`, and its User-Agent to `python-httpx2/<version>` — [httpx2 CHANGELOG 2.0.0b1](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)

**OpenTelemetry GenAI semantic conventions: where they live and their status**
- The opentelemetry.io GenAI spans and metrics pages now say the conventions moved to `https://github.com/open-telemetry/semantic-conventions-genai` and are "no longer maintained" there — [opentelemetry.io gen-ai spans page](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- The new repo extends the core conventions via Weaver. Its schema URL is still "TODO", and no releases are published. The GenAI README status is **Development** — [semantic-conventions-genai repo](https://github.com/open-telemetry/semantic-conventions-genai); [GenAI README](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md)

**OTel GenAI span (`gen_ai.client.inference`, Development)**
- The span name SHOULD be `{gen_ai.operation.name} {gen_ai.request.model}`.
- **Required**: `gen_ai.operation.name` (for example `chat`) and `gen_ai.provider.name` (for example `openai`).
- **Conditionally required**: `error.type` (Stable; on error) and `gen_ai.request.model` (if available).
- **Recommended**: `server.address`/`server.port`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, and `gen_ai.response.time_to_first_chunk` (streaming, in seconds).
- Usage attributes:
  - `gen_ai.usage.input_tokens` "SHOULD include cached tokens and input tokens of all modalities".
  - `gen_ai.usage.output_tokens`.
  - `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` and `gen_ai.usage.reasoning.output_tokens`, each recommended "when available and non-zero".
- **Opt-In content attributes**: `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` and `gen_ai.tool.definitions`, plus `*_ref` URIs for external storage.
- Source: [GenAI client inference conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)

**OTel GenAI metrics (all Development)**
- `gen_ai.client.inference.duration` (Histogram, s), `gen_ai.client.inference.time_to_first_chunk` (Histogram, s) and `gen_ai.client.inference.time_per_output_chunk` (Histogram, s) — [GenAI client inference conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)
- Token counters `gen_ai.client.inference.usage.input_tokens` / `.output_tokens` / `.cache_read.input_tokens` / `.cache_write.input_tokens` / `.reasoning.output_tokens` (Counter, `{token}`), plus per-operation histograms `gen_ai.client.inference.operation.input_tokens` / `.output_tokens`.
  - "When systems report both used tokens and billable tokens, instrumentation MUST report billable tokens".
  - Instrumentation should record contributions from failed operations too, and set `gen_ai.token.modality` to `unknown` when the modality is unknown.
  - Source: [GenAI token metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md)
- **Inconsistency inside the new repo**: the general `gen-ai-metrics.md` still lists `gen_ai.client.operation.duration` (the older name), while `client-inference.md` defines `gen_ai.client.inference.duration` — [gen-ai-metrics.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) vs [client-inference.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)

**Existing third-party instrumentation**
- Logfire (Pydantic) instruments httpx2 clients with `logfire.instrument_httpx()`, scoped per client if desired. `capture_all=True` also records headers and bodies — [httpx2 README](https://github.com/pydantic/httpx2)

### Inferences
**Logging levels**
- DEBUG: per-attempt `provider/model/status/latency/attempt#/cooldown_until`.
- INFO: sparingly, for example once on failover.
- WARNING: a key was disabled (401/403), or all candidates are exhausted.
- Never log request or response bodies by default.
- Redact `Authorization`, `x-api-key`, `api-key` and `?key=` query parameters (Gemini-style) in every log line, exception message and `__repr__`.
- Because `capture_all=True`-style HTTP instrumentation records headers, document that users enabling it must scrub auth headers.

**Hooks as the stable API**

```python
@dataclass(frozen=True, slots=True)
class AttemptEvent: provider: str; model: str; attempt: int; started_at: float
@dataclass(frozen=True, slots=True)
class AttemptResult: provider: str; model: str; status: int|None; error_type: str|None; latency_s: float; ttft_s: float|None; usage: Usage|None; cooldown_until: float|None
class Hooks(Protocol):
    def on_attempt_start(self, e: AttemptEvent) -> None: ...
    def on_attempt_end(self, r: AttemptResult) -> None: ...
    def on_failover(self, from_: AttemptResult, to: AttemptEvent) -> None: ...
```

- Hooks must be exception-safe (catch and log), and must be cheap or sync-only (no `await` inside the core lock).
- An optional `freeai[otel]` extra would depend on `opentelemetry-api` only, never the SDK. It would emit one span per **attempt** (`chat {model}`, `gen_ai.provider.name`, `error.type`) under a parent span per logical call, plus the duration, TTFT and token metrics.
- Because the conventions are Development and names changed even in 2026, pin the attribute names in one module, label the integration experimental, and avoid exposing the OTel names in the public API.

**Usage and cost**
- Return a `Usage(input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, estimated: bool)` on every response, and on the final stream event.
- Expose a `client.stats()` snapshot: per-provider calls, failures, current cooldowns, RPD used vs limit, and remaining quota from headers.
- For free tiers, "cost" is mostly **quota consumption** rather than dollars. Report `requests_today/limit` and `tokens_this_minute/limit` per provider, plus an optional price table for paid fallbacks.

### Gaps
- The new repo had no mention of `OTEL_SEMCONV_STABILITY_OPT_IN` or `gen_ai_latest_experimental` on the pages fetched, and no stabilization timeline. Whether the opt-in mechanism still applies to GenAI after the move is unverified.
- I did not verify whether `gen_ai.system` is formally deprecated in the new registry (`docs/registry/attributes/gen-ai.md` was not fetched). The current client-inference page uses only `gen_ai.provider.name`.
- I did not review callback-API designs in LiteLLM or LangChain in primary docs.

## 7. API ergonomics: OpenAI-compatible responses vs custom dataclasses, typed exceptions, Pydantic vs dataclasses, env-var configuration, `py.typed` and strict typing

### Takeaway
For a dependency-light SDK, use **stdlib `@dataclass(frozen=True, slots=True)` response types shaped like OpenAI's chat-completion fields** (`choices[0].message.content`, `usage.prompt_tokens`…), keep the raw provider JSON on `.raw`, and make **Pydantic unnecessary**. Ship a shallow **typed exception hierarchy** that carries per-attempt diagnostics. Ship `py.typed`, run a strict type checker in CI, and read configuration from **explicit args > `FREEAI_*` env > provider-native env vars**.

### Cited Findings
- openai-python's runtime dependencies are `httpx2>=2.12.0,<3`, `pydantic>=1.10.13,<3` (excluding 2.0–2.3), `typing-extensions`, `anyio`, `sniffio` and `jiter`. It requires Python ≥3.10, uses the classifier "Typing :: Typed" and the license `Apache-2.0`. Its dev tooling uses `ty==0.0.92` — [openai-python pyproject.toml](https://github.com/openai/openai-python/blob/main/pyproject.toml)
- openai-python errors:
  - All errors inherit from `openai.APIError`.
  - Connection or timeout failures raise subclasses of `openai.APIConnectionError` (for example `APITimeoutError`).
  - Non-success status codes raise `openai.APIStatusError` with `status_code` and `response` properties.
  - Source: [openai-python README](https://github.com/openai/openai-python/blob/main/README.md)
- openai-python validates `max_retries` strictly: `None` → `TypeError` ("Use 0 to disable retries or a large integer…"), and a negative value → `ValueError` — [openai `_base_client.py`](https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py)
- uv's build backend expects the module at `src/<package_name>/__init__.py` by default (configurable via `module-name`/`module-root`). Wheels include the module root. The page does not mention `py.typed` explicitly — [uv build backend docs](https://docs.astral.sh/uv/concepts/build-backend/)

### Inferences
**Response shape**
- Mimic the OpenAI ChatCompletion field names, since most free endpoints are OpenAI-compatible. That lets users swap from `openai` with minimal friction. Add `provider`, `model`, `attempts`, `usage.estimated` and `.raw`.
- Avoid inheriting OpenAI's full surface (tools, logprobs…) until needed; keep unknown fields reachable in `.raw`.
- A `to_dict()` helper is enough; Pydantic adds a compiled dependency (`pydantic-core`) for validation the SDK doesn't need, since provider JSON is normalized by hand.

**Exceptions**

```
FreeAIError
├── ConfigError                # missing keys, unknown model alias
├── RequestError               # 400-class caller errors, not retried/failover-able (bad params, context too long)
├── AllCandidatesFailed        # .attempts: list[AttemptResult]; .earliest_retry_at
│   └── QuotaExhausted         # every candidate cooling down / out of RPD; .retry_at
├── ProviderError              # single-attempt error surfaced when failover disabled; .provider .model .status .body (redacted)
├── TimeoutError (FreeAITimeout) # deadline exceeded
└── StreamInterrupted          # .partial_text, after first chunk
```

- Keep `__cause__` set to the underlying httpx exception, as openai does.

**Configuration**
- Precedence: explicit args, then `FREEAI_CONFIG` (TOML path), `FREEAI_STATE` (`memory`/`sqlite:///path`/`redis://…`), `FREEAI_LOG_LEVEL`, then provider keys from conventional names (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, …).
- Never echo key values. Show only the last 4 characters in diagnostics.

**Typing**
- Put `src/freeai/py.typed` (an empty file) in the package. PEP 561 is the governing spec, but it was not fetched this session.
- Run `mypy --strict` or `pyright --strict` (or `ty`, which openai uses) in CI.
- Use `Literal` for roles and finish reasons, `TypedDict` for message input dicts (users pass plain dicts), and `@overload` so `chat(stream=True)` returns a stream type.

### Gaps
- I fetched no primary benchmarks of dataclasses vs Pydantic v2 vs msgspec construction cost. The recommendation rests on dependency weight, not measured speed.
- I did not open PEP 561 to cite it, and did not confirm that uv_build includes `py.typed` automatically. It should, since it lives inside the module directory, but verify with `uv build` and an `unzip -l` of the wheel.

## 8. Testing: `httpx.MockTransport` vs respx vs VCR/pytest-recording, property-based scheduler tests, live contract tests in CI with secrets, fake clocks

### Takeaway
Keep **`MockTransport` as the main unit-test tool**: it needs no dependency and survives an httpx→httpx2 migration, since the API is identical. Add **Hypothesis stateful tests** for the router, limiter and cooldown core with an **injected fake clock**. Use **VCR cassettes** (vcrpy/pytest-recording) only for a small set of recorded provider fixtures, with header filtering. Run **live contract tests** as a separate opt-in CI job using repository secrets, tolerant of free-tier flakiness. respx and pytest-httpx are tied to classic `httpx`.

### Cited Findings
- `httpx.MockTransport()` was added to httpx years ago ("allowing to mock out a transport using pre-determined responses"). httpx2 "is a fork of httpx 0.28.1. It has the same public API", and the two packages "can be installed in the same environment, at the same time" — [httpx2 CHANGELOG](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md); [httpx2 migration guide](https://github.com/pydantic/httpx2/blob/main/docs/migration.md)
- Dependency metadata (current PyPI):

  | Package | Version | HTTP-client dependency or note |
  |---|---|---|
  | respx | 0.23.1 | `httpx>=0.25.0` |
  | pytest-httpx | 0.36.2 | pins `httpx==0.28.*` |
  | vcrpy | 9.4.2 | lists both `httpx` and `httpx2` in its test extras |
  | pytest-recording | 0.14.1 | depends on `vcrpy>=2.0.1` |

  Sources: [respx PyPI](https://pypi.org/project/respx/); [pytest-httpx PyPI](https://pypi.org/project/pytest-httpx/); [vcrpy PyPI](https://pypi.org/project/vcrpy/); [pytest-recording PyPI](https://pypi.org/project/pytest-recording/)
- The OpenAI README notes that custom "request mock[s]" are affected by the HTTPX2 migration and points to its migration guide — [openai-python README](https://github.com/openai/openai-python/blob/main/README.md)
- Hypothesis stateful testing uses `RuleBasedStateMachine` with `@rule` (with `Bundle` targets), `@invariant`, `@precondition` and `@initialize` to generate sequences of operations — [Hypothesis stateful testing](https://hypothesis.readthedocs.io/en/latest/stateful.html)
- PyrateLimiter's design makes `Clock` a pluggable dependency that "only needs `now() -> int` (sync or async)" — [PyrateLimiter README](https://github.com/vutran1710/PyrateLimiter)

### Inferences
**Test layers**
1. **Pure core** (no transport): table tests plus Hypothesis.
   - Use a `RuleBasedStateMachine` with rules such as `send(candidate)`, `respond(status, headers)`, `advance_clock(dt)` and `new_day()`.
   - Invariants:
     - never select an endpoint inside its cooldown;
     - GCRA never admits more than `limit + burst` per window;
     - RPD count never exceeds the limit within one period;
     - the retry budget is never negative;
     - total sleep never exceeds the deadline.
   - Inject `clock: Callable[[], float]` and `sleep: Callable[[float], None]` (async: `Callable[[float], Awaitable[None]]`), and have fake sleep advance the fake clock. This avoids freezegun or time-machine patching of `time.monotonic`.
2. **Transport level**: `MockTransport` handlers that script per-host behavior.
   - Cover: 429 with `retry-after`, a 200 SSE body with chunk splits at awkward offsets, a mid-stream `event: error`, a JSON 200 with an error body, and a connection reset (raise `httpx.ConnectError` from the handler).
   - Parametrize SSE fixtures through a byte-chunker that splits at every offset, which catches CRLF-boundary bugs.
3. **Recorded fixtures** (optional): vcrpy cassettes with `filter_headers=["authorization","x-api-key"]` and `filter_query_parameters=["key"]`.
   - Review cassettes in PRs; they act as regression fixtures for provider JSON shapes.
   - Refresh them deliberately, because free providers change formats.
4. **Live contract tests**: mark them `@pytest.mark.live` and skip them unless `FREEAI_LIVE=1` and keys are present.
   - Run on schedule (nightly) and on manual dispatch, not on every PR.
   - Use tiny prompts (`max_tokens=5`) to save quota.
   - Assert the shape (non-empty content, usage present or estimated, SSE terminates) rather than the text.
   - Treat 429 or quota exhaustion as `xfail`/skip, not a failure.

**Dependency direction**
- If `freeai` moves to httpx2, respx and pytest-httpx would not intercept httpx2 clients, since they patch or depend on `httpx`. That is another reason to standardize on `MockTransport`.

### Gaps
- I did not verify GitHub's rule that secrets are not exposed to workflows from forked PRs, or the `pull_request_target` caveats, in GitHub docs this session.
- I did not verify whether respx works with httpx2 through some shim. The metadata only shows a dependency on `httpx`.
- I did not fetch documentation for `time-machine` or `freezegun` handling of `time.monotonic`.

## 9. Packaging and release: pyproject with the uv build backend, PyPI Trusted Publishing via GitHub Actions, SemVer, supported Python versions, license, security/redaction

### Takeaway
- Build with `uv_build` (pure Python; pin `<0.14`).
- Declare a PEP 639 SPDX `license` and `license-files`.
- Publish from a tag-triggered GitHub Actions workflow using **Trusted Publishing** (`id-token: write`, a `pypi` environment). `pypa/gh-action-pypi-publish` **generates PEP 740 attestations by default**; `uv publish` uploads attestations if present but does not generate them.
- Stay on **0.y.z SemVer** until the API settles.
- Support **Python ≥3.11 (or ≥3.12)**: 3.10 reached end-of-life on 2026-10-01, and httpx2 has dropped 3.10.

### Cited Findings
**uv build backend**
- Config: `[build-system] requires = ["uv_build>=0.13.7,<0.14"]`, `build-backend = "uv_build"`. The page recommends an upper bound because of uv's versioning policy. The backend "currently only supports pure Python code".
- The default module location is `src/<package_name>/__init__.py`.
- `project.license-files` is copied into `.dist-info`, and patterns use PEP 639's portable glob syntax.
- Source: [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)

**uv publish**
- `uv build` produces sdist and wheel. With `uv publish` "from GitHub Actions or another Trusted Publisher, you don't need to set any credentials", and uv attempts to invalidate the short-lived token afterwards.
- uv uploads attestations by default (`--no-attestations` or `UV_PUBLISH_NO_ATTESTATIONS` disables this), but "uv publish does not currently generate attestations".
- Source: [uv guide: building and publishing a package](https://docs.astral.sh/uv/guides/package/)

**PyPI Trusted Publishing**
- `permissions: id-token: write` is "mandatory for Trusted Publishing". A GitHub environment named `pypi` is "optional, but strongly encouraged". The workflow uses `pypa/gh-action-pypi-publish@release/v1` with no tokens. twine also supports the flow — [PyPI docs: Using a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- gh-action-pypi-publish generates and uploads signed digital attestations for all distribution files by default when using Trusted Publishing (disable with `attestations: false`). Trusted Publishing "cannot be used from within a reusable workflow"; the publish step must run in a separate job of a non-reusable workflow — [pypa/gh-action-pypi-publish README](https://github.com/pypa/gh-action-pypi-publish)

**License metadata**
- PEP 639: `license` is an SPDX expression (for example `license = "MIT"` or `"MIT AND (Apache-2.0 OR BSD-2-Clause)"`), and `license-files` is a list of glob patterns — [Python Packaging User Guide: Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

**Supported Python versions**
- Python versions as of today:

  | Version | Status | End of life |
  |---|---|---|
  | 3.9 | EOL | 2025-10-31 |
  | 3.10 | EOL | 2026-10-01 |
  | 3.11 | security | 2027-10 |
  | 3.12 | security | 2028-10 |
  | 3.13 | security | 2029-10 |
  | 3.14 | bugfix | 2030-10 |
  | 3.15 | bugfix (released 2026-10-09) | 2031-10 |

  Source: [Python devguide: Status of Python versions](https://devguide.python.org/versions/)
- httpx2 dropped Python 3.9 in 2.1.0 (2026-05-15) and Python 3.10 in 2.15.0 (2026-10-19). It added 3.15 support in 2.10.0 — [httpx2 CHANGELOG](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)
- openai-python still declares `requires-python >= 3.10` with classifiers for 3.10–3.14 — [openai-python pyproject.toml](https://github.com/openai/openai-python/blob/main/pyproject.toml)

**Versioning and licenses**
- SemVer: "Major version zero (0.y.z) is for initial development. Anything MAY change at any time" — [semver.org](https://semver.org/)
- openai-python uses `license = "Apache-2.0"` — [openai-python pyproject.toml](https://github.com/openai/openai-python/blob/main/pyproject.toml)

**HTTP-stack dependency signals**
- httpx2 renamed `httpcore` → `httpcore2` (vendored in the same repo). It switched default certificate verification from `certifi` to `truststore`, which loads the OS trust store, in 2.3.0 — [httpx2 CHANGELOG](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)

### Inferences
**Recommended `pyproject.toml` skeleton**

```toml
[project]
name = "freeai"
version = "0.x.y"
requires-python = ">=3.12"   # or >=3.11 for reach; 3.10 is EOL and httpx2 already dropped it
license = "MIT"               # or "Apache-2.0" (explicit patent grant; what openai-python uses)
license-files = ["LICEN[CS]E*"]
dependencies = ["httpx>=0.27,<1"]  # or "httpx2>=2.12,<3" — see decision below
classifiers = ["Typing :: Typed", "Programming Language :: Python :: 3.12", "Programming Language :: Python :: 3.13", "Programming Language :: Python :: 3.14"]
[project.optional-dependencies]
redis = ["redis>=5"]
otel = ["opentelemetry-api>=1.30"]
tokens = ["tiktoken>=0.8"]
[build-system]
requires = ["uv_build>=0.13.7,<0.14"]
build-backend = "uv_build"
```

The version bounds for `redis`, `opentelemetry-api` and `tiktoken` above are illustrative and were not verified this session.

**httpx vs httpx2 decision**
- `httpx` 0.28.1 still works but has had no stable release since 2024-12. The two largest LLM SDKs now default to `httpx2`, and httpx2 gets security fixes and native SSE.
- Option A: keep `httpx` now, isolate it behind one internal `_http.py` module, and use `MockTransport`, which works in both.
- Option B: move to `httpx2` (the change is an import rename) and drop the custom SSE parser.
- Supporting both via `try: import httpx2 as httpx except ImportError: import httpx` is possible but doubles the test matrix.

**Release workflow**
- Trigger on `v*` tags.
- Job 1: `uv build`, then `uv run --isolated --no-project --with dist/*.whl python -c "import freeai"`, which smoke-tests the wheel.
- Job 2 (`environment: pypi`, `permissions: id-token: write`): download the artifact, then run `pypa/gh-action-pypi-publish@release/v1`. Use this action rather than `uv publish`, because it generates attestations.
- Publish to TestPyPI first, from the same workflow, for release candidates.

**Python support policy**
- Support every CPython version that is in bugfix or security status *and* supported by the HTTP dependency. Drop a version within one minor release after its EOL.
- Test 3.12–3.14 in CI, and add 3.15 now that it has been released.

**Secret hygiene**
- Store keys as `SecretStr`-like wrappers whose `__repr__` and `__str__` mask the value.
- Strip auth headers and `key=` query parameters from exception messages, logs and VCR cassettes.
- Never include full request headers in `AllCandidatesFailed` messages.
- Add a test that serializes every exception and log record with a fake key and asserts the key is absent.
- Enable GitHub secret scanning and Dependabot.

### Gaps
- PyPI's trusted-publisher page did not mention attestations, so the "default on" claim comes from the gh-action README alone.
- I did not fetch the uv versioning policy, so I can't say when the 0.14 build backend will be released or whether it will break compatibility.
- I fetched no primary source comparing MIT and Apache-2.0 for SDKs. The patent-grant point is general knowledge, not cited here.
- I did not research whether PyPI requires or recommends specific metadata for SBOMs (PEP 770) as of Nov 2026.
