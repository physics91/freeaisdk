# Existing presets: Groq, Cerebras, Gemini API, Mistral (state as of 2026-11-06)

Method note: the official docs were fetched as raw markdown on 2026-11-06 (Groq `console.groq.com/docs/*.md`, Cerebras `inference-docs.cerebras.ai/*.md`, Google `ai.google.dev/gemini-api/docs/*.md.txt`, Mistral `docs.mistral.ai/*.md`). The canonical page URL is cited below. "Live probe 2026-11-06" means an unauthenticated request with the fake key `invalid-key-probe`. No real API keys were available (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY` were all unset), so no real 429 or success response was observed on any provider. The Gemini probe failed: DNS resolution of `generativelanguage.googleapis.com` timed out from this environment.

**Headline findings that break preset assumptions:**
1. **Cerebras has no permanent free tier.** It offers a $5, 30-day Free Trial and requires a verified payment method. — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
2. **Mistral's official docs say Free mode includes no API access.** The free "Experiment" API plan appears to be discontinued. — [Mistral Usage and limits](https://docs.mistral.ai/admin/billing-usage/usage-limits)
3. **Groq's free text models are now only `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b` and `qwen/qwen3.8-27b`.** `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` are Enterprise-only. — [Groq Rate Limits](https://console.groq.com/docs/rate-limits), [Groq Supported Models](https://console.groq.com/docs/models)
4. **Google no longer publishes Gemini free-tier numbers in its docs.** The only numbers found come from a third-party source, partly backed by a forum report. — [Gemini Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

---

## Groq: base URL, `/models`, and model ID format

### Takeaway
`https://api.groq.com/openai/v1` is correct, and `GET /models` on it is documented. IDs are bare strings, some with an org prefix (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`). Chat requests use those same IDs. The list covers all active models, including speech, TTS and guard models, so the SDK must filter it.

### Cited Findings
- The OpenAI-client config uses `base_url="https://api.groq.com/openai/v1"` with `GROQ_API_KEY`. — [Groq OpenAI Compatibility](https://console.groq.com/docs/openai)
- "You can use the `https://api.groq.com/openai/v1/models` endpoint to return a JSON list of all active models." Production model IDs: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `whisper-large-v3`, `whisper-large-v3-turbo`. Preview IDs: `canopylabs/orpheus-arabic-saudi`, `canopylabs/orpheus-v1-english`, `meta-llama/llama-prompt-guard-2-22m`, `meta-llama/llama-prompt-guard-2-86m`, `meta-models/muse-spark-open`, `minimaxai/minimax-m2.7`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`. — [Groq Supported Models](https://console.groq.com/docs/models)
- Live probe 2026-11-06 of `GET https://api.groq.com/openai/v1/models` and `POST .../chat/completions` with a bad key: `HTTP/2 401` and body `{"error":{"message":"Invalid API Key","type":"invalid_request_error","code":"invalid_api_key"}}`. Response headers included `x-groq-region: dls` and `x-request-id: req_...` (chat only). — live probe of [api.groq.com/openai/v1/models](https://api.groq.com/openai/v1/models)

### Inferences
- `GET /models` returns non-chat models: Whisper (speech-to-text), Orpheus (TTS) and prompt-guard classifiers. It also returns Enterprise-only models that free keys cannot call (see the next section). An unfiltered list would rotate into models that return 4xx. Use an allowlist or filter by ID pattern.

### Gaps
- The exact `/models` JSON fields (for example `active`, `context_window`, `owned_by`) were not seen on the current docs page or in a live authenticated call.
- It is unverified whether `/models` hides Enterprise-only models from non-Enterprise orgs.

## Groq: free-tier models, limits, scope, plans, and recent changes

### Takeaway
Free-tier chat models are `openai/gpt-oss-120b` and `openai/gpt-oss-20b` (30 RPM / 1K RPD / 8K TPM / 200K TPD), `qwen/qwen3.8-27b` (same limits) and `openai/gpt-oss-safeguard-20b` (3 RPM / 1K RPD / 2K TPM / 200K TPD). Limits are set per model and enforced per **organization**. Llama 3.x is now Enterprise-only.

### Cited Findings
- Limit units: RPM, RPD, TPM, TPD, ASH, ASD, ITPM, OTPM. "Rate limits apply at the organization level, not individual users. You can hit any limit type depending on which threshold you reach first." "Cached tokens do not count towards your rate limits." Some orgs also have separate ITPM/OTPM caps, shown as "X in / Y out" on the Limits page. — [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- Rate-limit table (columns: MODEL ID | RPM | RPD | TPM | TPD | ASH | ASD): `openai/gpt-oss-120b` 30 / 1K / 8K / 200K. `openai/gpt-oss-20b` 30 / 1K / 8K / 200K. `openai/gpt-oss-safeguard-20b` 3 / 1K / 2K / 200K. `qwen/qwen3.8-27b` 30 / 1K / 8K / 200K. `meta-llama/llama-prompt-guard-2-22m` and `-86m` 30 / 14.4K / 15K / 500K. `canopylabs/orpheus-*` 10 / 100 / 1.2K / 3.6K. `whisper-large-v3` and `-turbo` 20 RPM / 2K RPD / 7.2K ASH / 28.8K ASD. — [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- **Conflict:** the same page says "Note that the limits shown below are the base limits for the Developer plan". The models page lists much higher "RATE LIMITS (DEVELOPER PLAN)": `openai/gpt-oss-120b` "250K TPM 1K RPM", `qwen/qwen3.8-27b` "250K TPM 1K RPM", `openai/gpt-oss-safeguard-20b` "150K TPM 1K RPM". — [Groq Rate Limits](https://console.groq.com/docs/rate-limits) vs [Groq Supported Models](https://console.groq.com/docs/models)
- `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `meta-models/muse-spark-open` and `minimaxai/minimax-m2.7` are labeled "Enterprise" with price and limits "ContactSales". They are absent from the rate-limits table. — [Groq Supported Models](https://console.groq.com/docs/models)
- Context and max output: `openai/gpt-oss-120b` and `-20b` 131,072 context / 65,536 max completion. `qwen/qwen3.8-27b` 131,072 / 16,384, with "MAX FILE SIZE 20 MB". `llama-3.3-70b-versatile` 131,072 / 32,768. — [Groq Supported Models](https://console.groq.com/docs/models)
- The upgrade path is the "Developer plan", which adds higher limits, Batch and Flex processing. — [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- An older changelog entry (undated in my extract; it appears under the Remote MCP launch) lists `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`, `moonshotai/kimi-k2-instruct-0905` and the Llama 4 models as tool-use models. These were public models in 2025. Recent changelog entries add models as Enterprise-only, for example "`minimaxai/minimax-m2.5` and `qwen/qwen3-vl-32b-instruct` are now available on GroqCloud for Enterprise customers". — [Groq Changelog](https://console.groq.com/docs/changelog)

### Inferences
- The rate-limits table's numbers (30 RPM / 8K TPM) match Groq's historical free tier and are far below the Developer numbers on the models page. Most likely the table shows the **free** tier and the "Developer plan" sentence is a doc error. Treat it as ambiguous anyway: the console Limits page is authoritative per org.
- Limits are set per model but enforced per org, so per-model benching fits. Several keys under one org share one bucket, so rotating keys inside an org gains nothing.
- At 8K TPM, one request with a large prompt plus `max_tokens` can exceed the TPM bucket. The SDK should size requests below TPM or expect immediate token 429s.

### Gaps
- No date was found for when Llama 3.x moved to Enterprise-only.
- Whether free-tier orgs get the ITPM/OTPM split is not stated.
- No official statement was found on phone verification at sign-up.

## Groq: 429 shape, other error codes, and rate-limit headers

### Takeaway
A 429 carries `retry-after` (seconds) and a JSON body with `error.type` (observed: `"tokens"`) and `error.code: "rate_limit_exceeded"`. The message names the limit (TPM, RPD and so on) and gives "Please try again in Xs". Every response carries `x-ratelimit-*` headers, but **the request headers are per day (RPD) and the token headers are per minute (TPM)**.

### Cited Findings
- Headers, with the docs' illustrative values: `retry-after: 2` (seconds). `x-ratelimit-limit-requests: 14400`, "Always refers to Requests Per Day (RPD)". `x-ratelimit-limit-tokens: 18000`, "Always refers to Tokens Per Minute (TPM)". `x-ratelimit-remaining-requests: 14370` (RPD). `x-ratelimit-remaining-tokens: 17997` (TPM). `x-ratelimit-reset-requests: 2m59.56s` (RPD). `x-ratelimit-reset-tokens: 7.66s` (TPM). "`retry-after` is only set if you hit the rate limit and status code 429 is returned. The other headers are always included." — [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- Observed 429 body (LiteLLM via Aider, 2025-02-16): `{"error":{"message":"Rate limit reached for model `gemma2-9b-it` in organization `...` service tier `on_demand` on tokens per minute (TPM): Limit 15000, Used 11972, Requested 4351. Please try again in 5.289s. Visit https://console.groq.com/docs/rate-limits for more information.","type":"tokens","code":"rate_limit_exceeded"}}`. — [Aider issue #3265 (2025-02)](https://github.com/Aider-AI/aider/issues/3265)
- Error envelope: `{"error": {"message": "...", "type": "invalid_request_error"}}`. Codes: 400, 401, 403, 404, 413 (request too large), 422, 424 (Remote MCP failed dependency), 429, **498 "Flex Tier Capacity Exceeded"**, 499 (request cancelled, seen in logs), 500, 502, 503. "You will not be charged for requests that return server error codes." — [Groq Errors](https://console.groq.com/docs/errors)

### Inferences
- Classification hints: the message substrings "tokens per minute (TPM)", "requests per day (RPD)" and "tokens per day (TPD)" identify the bucket. `error.type` was `"tokens"` for TPM. A `"requests"` value for RPM/RPD hits is plausible but unverified.
- For proactive throttling, `x-ratelimit-remaining-requests == 0` means **the daily quota is gone**, so bench until `x-ratelimit-reset-requests`. Do not read it as "wait a minute". A low `x-ratelimit-remaining-tokens` is minute-scale.
- The SDK's Groq-duration parser (`2m59.56s`, `7.66s`) fits these headers. The "Please try again in 5.289s" message text is a usable fallback.
- 498 only applies to Flex (paid). Treat 503/500/502 as transient and do not bench long.

### Gaps
- No primary-source example was found of an RPD or RPM 429 body, or of the exact `type` values beyond `"tokens"`.
- Whether `x-ratelimit-*` headers appear on streaming responses was not verified.

## Groq: OpenAI-compatibility quirks

### Takeaway
Groq mostly matches OpenAI. Known 400 triggers: `logprobs`, `logit_bias`, `top_logprobs`, `messages[].name`, `n != 1`, and `reasoning_format: "raw"` combined with JSON mode or tools. Reasoning comes back in `message.reasoning`, not `reasoning_content`. Structured Outputs cannot be combined with streaming or tool use.

### Cited Findings
- "The following fields are currently not supported and will result in a 400 error ... `logprobs`, `logit_bias`, `top_logprobs`, `messages[].name`. If `N` is supplied, it must be equal to 1." `temperature: 0` is converted to `1e-8`. — [Groq OpenAI Compatibility](https://console.groq.com/docs/openai)
- `reasoning_format` takes `parsed`, `raw` or `hidden`. `parsed` "separates reasoning into a dedicated message.reasoning field". `raw` puts reasoning in `<think>` tags inside the content. "If reasoning is explicitly set to `raw` with JSON mode or tool use enabled, we will return a 400 error." `include_reasoning` (default true) and `reasoning_format` are mutually exclusive. `reasoning_effort` works on GPT-OSS 20B/120B and Qwen 3.8 27B. — [Groq Reasoning](https://console.groq.com/docs/reasoning)
- Structured Outputs: `response_format: {"type":"json_schema","json_schema":{...,"strict":true}}`. Strict (constrained decoding) supports `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` and `meta-models/muse-spark-open`. Best-effort (`strict:false`) also supports `openai/gpt-oss-safeguard-20b`. "Streaming and tool use are not currently supported with Structured Outputs." Strict mode requires every field to be `required` and `additionalProperties: false`. — [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs)
- Groq also offers a Responses API. — [Groq OpenAI Compatibility](https://console.groq.com/docs/openai)
- The chat API reference lists `stream_options` ("Options for streaming response. Only set this when you set `stream: true`"), but the snapshot does not expand its properties. It also documents an `x_groq` object as "Groq-specific metadata for non-streaming chat completion responses", plus `usage` and `usage_breakdown`. — [Groq API Reference](https://console.groq.com/docs/api-reference)

### Inferences
- The SDK should drop or guard `logprobs`, `logit_bias`, `top_logprobs`, `name` and `n>1` for Groq, or it will get 400s that are not rate limits.
- A normalized "reasoning" accessor should read `message.reasoning` (Groq and Cerebras) as well as `reasoning_content`.

### Gaps
- Not verified: whether `stream_options.include_usage` is honored and in which chunk streamed usage appears. The parameter exists in the reference, but its sub-properties were not captured.
- Not verified: `tool_choice: "required"` and named tool choice, `parallel_tool_calls`, and the image-input format for `qwen/qwen3.8-27b`.

## Groq: data policy, ToS, and region

### Takeaway
By default Groq keeps no inference inputs or outputs. It may log them for up to 30 days for reliability or abuse checks, and it names "rate-limit circumvention" as suspected abuse. No training-use clause and no region list were found.

### Cited Findings
- "By default, Groq does not retain customer data for inference requests." Inputs and outputs may be logged temporarily "only when: Troubleshooting errors ... or Investigating suspected abuse (e.g. rate-limit circumvention). These logs are retained for up to 30 days." Users can opt out in Data Controls. Inference endpoints are ZDR-eligible. — [Your Data in GroqCloud](https://console.groq.com/docs/your-data)

### Inferences
- Rotating several accounts or orgs to evade Groq limits risks abuse flagging, since Groq names rate-limit circumvention explicitly. Rotating across *models* inside one org uses limits as designed.

### Gaps
- The Groq Terms of Use (https://groq.com/terms-of-use) were not fetched. Clauses on multiple accounts, commercial use on the free tier, regional restrictions (including Korea) and sign-up phone verification are unverified.

---

## Cerebras: base URL, `/models`, and model ID format

### Takeaway
`https://api.cerebras.ai/v1` is correct, and `GET /v1/models` returns a standard OpenAI list with bare IDs (`gpt-oss-120b`, `qwen-3.8-27b`). **Cerebras error bodies are flat, with no `error` wrapper.**

### Cited Findings
- Base URL `https://api.cerebras.ai/v1`; the full endpoint is `https://api.cerebras.ai/v1/chat/completions`. The `model` field takes an exact Cerebras model ID (Shared Inference) or an endpoint ID (Dedicated). — [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- `GET https://api.cerebras.ai/v1/models` example response: `{"object":"list","data":[{"id":"gpt-oss-120b","object":"model","created":0,"owned_by":"Cerebras"},{"id":"qwen-3.8-27b","object":"model","created":0,"owned_by":"Cerebras"}]}`. — [Cerebras List models](https://inference-docs.cerebras.ai/api-reference/models/list-models)
- Live probe 2026-11-06 with a bad key, on both `/v1/models` and `/v1/chat/completions`: `HTTP/2 401` and body `{"message":"Wrong API Key","type":"invalid_request_error","param":"api_key","code":"wrong_api_key"}`. There is no top-level `error` key. — live probe of [api.cerebras.ai/v1/models](https://api.cerebras.ai/v1/models)

### Inferences
- If `freeai` reads `body["error"]["message"]`, it will miss Cerebras errors. Parse both `body.error.*` and top-level `body.message/type/code`.

### Gaps
- Whether `/models` lists only models the org may call (for example, hides Dedicated-only models) was not verified live.

## Cerebras: free tier (now a time-limited trial), limits, scope, and recent changes

### Takeaway
**Cerebras no longer has a permanent free tier.** New accounts get $5 in credits that expire after 30 days, and only after adding a verified payment method. After that, API access stops until credits are bought. Free Trial limits per model (`gpt-oss-120b`, `qwen-3.8-27b`): 5 RPM, 30K uncached TPM, 90K total TPM, 1M TPH, 1M TPD. Limits apply per model, per organization and per project.

### Cited Findings
- FAQ: "Is there a permanently free tier? No. The Free Trial is time- and credit-bounded: $5 in credits that expire 30 days after they're granted ... Cerebras doesn't currently offer a no-cost tier that renews automatically or a per-model always-free allowance." "New accounts receive $5 in free credits after adding a verified payment method ... If you skip adding a payment method at sign-up, Playground and API access remain inactive until you do." "API and Playground access stop on the Free Trial tier until you purchase credits. Your API keys, projects, and settings remain intact." — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
- The change is dated **2026-07-16** ("Free Trial credits for new accounts"). The dual-bucket "Uncached Rate Limits" model was introduced the same day and was "fully in effect for all organizations on August 17, 2026". — [Cerebras Change Log](https://inference-docs.cerebras.ai/support/change-log)
- Free Trial table (Model | RPM | Uncached TPM | Total TPM | TPH | TPD): `gpt-oss-120b` 5 | 30K | 90K | 1M | 1M. `qwen-3.8-27b` 5 | 30K | 90K | 1M | 1M. Free Trial image limit: 2 per request, 10 MiB total payload. — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
- Developer (Pay as You Go): `gpt-oss-120b` 1M uncached TPM / 3M total / 1K RPM. `qwen-3.8-27b` 150K / 750K (temporarily raised from 450K) / 300 RPM. "Hourly and daily restrictions don't apply to Developer (Pay as You Go) tier users." Enterprise limits are provisioned per org. — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
- "Rate limits apply at the organization level, not the user level, and vary based on the model." Total TPM is 3x uncached by default, and both buckets are enforced on their own. Replenishment is a token bucket (continuous refill), not fixed resets. The token estimate is input plus `max_completion_tokens` (or an estimated upper bound), and "If this estimated token consumption would exceed your available token quota, the request is rate limited before processing begins." — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
- "Cerebras uses a two-level quota model: requests are checked against both the project-level limit and the org-level ceiling." — [Cerebras Usage & Monitoring](https://inference-docs.cerebras.ai/console/usage-monitoring)
- Shared Inference catalog (Model ID | Context free/paid): `gpt-oss-120b` 65k / 131k. `qwen-3.8-27b` 64k / 128k. Other model families are Dedicated-only. — [Cerebras Model Catalog](https://inference-docs.cerebras.ai/models/overview)
- Deprecations: `llama-3.3-70b` and `qwen-3-32b` (2026-02-16). `llama3.1-8b` and `qwen-3-235b-a22b-instruct-2507` (2026-05-27). `zai-glm-4.7` (2026-08-17). `gemma-4-31b` removed from Shared Inference (2026-09-03). Older entries: `qwen-3-coder-480b` (2025-11-05), `llama-4-scout-17b-16e-instruct` (2025-11-03) and others. — [Cerebras Deprecations](https://inference-docs.cerebras.ai/support/deprecation)

### Inferences
- In a "free models" rotation SDK, Cerebras should be a **trial or paid provider**, not a standing free member. After 30 days, or once $5 is spent, every call fails at the account level. That most likely shows up as HTTP 402 ("PaymentRequired" is in the status table), though the exact status after expiry was not verified.
- Old preset model names (`llama-3.3-70b`, `llama3.1-8b`, `qwen-3-32b`) are deprecated. Only `gpt-oss-120b` and `qwen-3.8-27b` remain on Shared Inference.
- With 5 RPM and pre-request token estimates, a large `max_completion_tokens` can trigger a 429 before any work is done. Send a modest `max_completion_tokens`.

### Gaps
- The exact status and body after trial credits expire or run out were not verified.
- No date was found for when the old permanent free tier (pre-July 2026) ended for *existing* accounts.

## Cerebras: 429 shape, other error codes, and rate-limit headers

### Takeaway
A 429 is documented and its message says whether the **uncached** or **total** token bucket was exceeded. The exact current 429 JSON and rate-limit header names are **not documented** in the current docs. The documented status set includes 402 and 503.

### Cited Findings
- Status table: 400 BadRequestError, 401 AuthenticationError, **402 PaymentRequired**, 403 PermissionDeniedError, 404 NotFoundError, 413 ContentTooLarge, 422 UnprocessableEntityError, 429 RateLimitError, 500 InternalServerError, 503 ServiceUnavailable. The SDK retries 408, 429 and >=500 twice by default. The default timeout is 60 s. — [Cerebras Error Codes](https://inference-docs.cerebras.ai/support/error)
- "If you exceed your rate limits, you will receive a 429 Too Many Requests error. The error message will indicate whether your **uncached** or **total** token limit was exceeded." — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)
- Image-input error strings, for example 413 `"Total request size exceeds maximum"` and `"Number of image inputs exceeds maximum"`, and 400 `"HTTPS image URLs are not supported. Use a base64-encoded data URI instead."` and `"Model {model} does not support image inputs"`. — [Cerebras Error Codes](https://inference-docs.cerebras.ai/support/error)
- A `queue_threshold` request header (50 to 20000 ms, Private Preview) makes `flex` and `auto` service-tier requests be "preemptively rejected if the rolling average queue time exceeds this threshold". — [Cerebras Chat Completions API](https://inference-docs.cerebras.ai/api-reference/chat-completions)
- Rate-limit headers: a search-engine snippet credited the Cerebras rate-limits page with saying custom rate-limit headers such as `x-ratelimit-limit-requests-day` are added to every response. The current page (fetched 2026-11-06) has no header section, and none of the downloaded Cerebras docs mention `x-ratelimit`. — [Cerebras Rate Limits (current)](https://inference-docs.cerebras.ai/support/rate-limits)

### Inferences
- Classify Cerebras 402 as account-level (credits gone), not model-level. Classify a 429 as model-level within the org or project, which matches the SDK's current assumption.
- Since refill is continuous, a short bench (seconds) usually suffices for minute buckets. On the Free Trial, hitting the 1M TPH or TPD caps needs a much longer bench.

### Gaps
- No primary source gives the current 429 JSON body, `code` strings (for example whether `token_quota_exceeded`, `request_quota_exceeded` or `queue_exceeded` still exist) or `retry-after` presence. Third-party pages were judged unreliable.
- Current rate-limit header names on success and 429 responses are unverified. A live authenticated `curl -D -` is needed.

## Cerebras: OpenAI-compatibility quirks

### Takeaway
Cerebras is compatible for common flows, with sharp edges. `n` must be 1. `gpt-oss-120b` rejects `tools` combined with `response_format`. Images must be base64 data URIs. Reasoning comes back in `message.reasoning` / `delta.reasoning`. `reasoning_format` and `clear_thinking` are non-standard and go via `extra_body`.

### Cited Findings
- Differences: "Only `n: 1` is supported". `service_tier` is "Not available with Shared Inference on the Developer Tier". For tool streaming, "Use standard streaming with `stream: true`. Do not send `tool_stream`." Image URLs must be "base64 PNG or JPEG data URI. External HTTPS image URLs are not supported." — [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- "`gpt-oss-120b` rejects requests containing both fields" (`tools` and `response_format`). — [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- Reasoning: `reasoning_effort` is standard. `qwen-3.8-27b` accepts `none`/`low`/`medium`/`high` and defaults to `high`. `reasoning_format` accepts `parsed`, `raw` or `hidden`. `parsed` "Returns reasoning separately in `choices[0].message.reasoning` or streaming `choices[0].delta.reasoning`". `gpt-oss-120b` supports parsed, raw and hidden. `qwen-3.8-27b` does not support `hidden`. `clear_thinking` controls whether historical reasoning is fed back. — [Cerebras Reasoning](https://inference-docs.cerebras.ai/capabilities/reasoning), [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- For `gpt-oss-120b`, `system` and `developer` roles both map to a developer-level instruction layer, so the same prompt may behave differently than on OpenAI. — [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- `max_tokens` is an alias for `max_completion_tokens`; "Do not send both parameters". The request body may be `application/vnd.msgpack` and/or gzip. Responses include a `time_info` object. — [Cerebras Chat Completions API](https://inference-docs.cerebras.ai/api-reference/chat-completions)
- The `X-Cerebras-Version-Patch: 2` header opts into API version 2: stricter structured-output schema validation, stricter multi-turn tool-call validation, and a separate `reasoning_logprobs`. — [Cerebras Change Log](https://inference-docs.cerebras.ai/support/change-log)

### Inferences
- The SDK should never send both `max_tokens` and `max_completion_tokens`, and should force `n=1`.
- When rotating a structured-output-plus-tools request onto Cerebras `gpt-oss-120b`, expect a 400. Treat it as "model not capable", not a rate limit.

### Gaps
- `stream_options` and `include_usage` do not appear in the Cerebras Chat Completions API reference fetched 2026-11-06 (grep found no hits), so support is undocumented. — [Cerebras Chat Completions API](https://inference-docs.cerebras.ai/api-reference/chat-completions)
- Not verified: `tool_choice: "required"` / named tool and `parallel_tool_calls` behavior (the chat API reference lists `parallel_tool_calls` and `tool_choice` parameters, but their values were not read).
- Max output tokens for `gpt-oss-120b` and `qwen-3.8-27b` on Cerebras were not captured. Only context lengths were.

## Cerebras: ToS, data policy, and region

### Takeaway
The Free Trial needs a verified payment method, which acts as the de facto identity check. ToS and privacy clauses were not checked.

### Cited Findings
- The Free Trial needs "a verified payment method"; without one, "Playground and API access remain inactive". — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits)

### Inferences
- Requiring a payment method ties each trial to a card identity. That is a practical gate against stacking accounts. Rotating across several trial accounts to stretch free credits would be impractical and likely abusive under the provider's terms (the terms themselves are unverified).

### Gaps
- Not fetched: the Cerebras ToS and privacy policy (training use of prompts, multi-account bans, regional availability including Korea, commercial use on the trial).

---

## Gemini API: base URL, `/models`, and model ID format

### Takeaway
`https://generativelanguage.googleapis.com/v1beta/openai` is correct. Google's docs write it with a trailing slash; `{base}/chat/completions` resolves to the same path. `GET {base}/models` and `GET {base}/models/{id}` are documented, and chat and retrieve use bare IDs such as `gemini-3.8-flash`. Whether list items have a `models/` prefix could not be checked.

### Cited Findings
- `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`. Chat endpoint: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`. List: `curl https://generativelanguage.googleapis.com/v1beta/openai/models -H "Authorization: Bearer GEMINI_API_KEY"`. Retrieve: `.../v1beta/openai/models/gemini-3.8-flash`, with Python `client.models.retrieve("gemini-3.8-flash")`. — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- "Support for the OpenAI libraries is still in beta while we extend feature support." — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- Live probe 2026-11-06 failed: `curl: (28) Resolving timed out` for `generativelanguage.googleapis.com` from this sandbox, so no headers or body were observed. — live probe of [generativelanguage.googleapis.com/v1beta/openai/models](https://generativelanguage.googleapis.com/v1beta/openai/models)

### Inferences
- Strip a leading `models/` from IDs returned by `/models` before sending them to chat. This costs nothing and guards against the historically reported prefix.
- `/models` will list many non-chat or non-free models (TTS, Live, image, embedding, `gemini-4-argon`, `gemini-3.1-pro-preview`, Omni). Use an allowlist of free text models (next section).

### Gaps
- The actual `/models` response (ID prefix, `owned_by`) is unverified. It needs a live call with a key from a network that can resolve `googleapis.com`.

## Gemini API: free-tier models, limits, scope, tiers, and recent changes

### Takeaway
Free-tier ("Free of charge") text models as of 2026-11-06 are `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, Gemma 4, and the deprecated `gemini-2.5-pro`, `gemini-2.5-flash` and `gemini-2.5-flash-lite`. Limits are **per project (not per API key), per model**, and RPD resets at midnight Pacific. Google no longer publishes the free numbers in its docs. A third-party source gives 5 RPM / 250K TPM / 20 RPD for 3.8 and 3.6 Flash and 10 RPM / 250K TPM / 500 RPD for 3.5 and 3.1 Flash-Lite.

### Cited Findings
- Free Tier column on the pricing page (Standard tab, input price), as extracted 2026-11-06. "Free of charge": `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, Gemma 4 (free tier only; "Not available" on paid), plus non-chat models (`gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`, `gemini-3.1-flash-tts-preview`, `gemini-3.8-live`, `gemini-3.5-transcribe`, `gemini-3.5-transcribe-live`, `gemini-3.5-live-translate-preview`, `gemini-2.5-flash-native-audio-preview-12-2025`, `gemini-2.5-flash-preview-tts`, `gemini-embedding-001`, `gemini-robotics-er-2-preview`). "Not available" on the free tier: `gemini-4-argon`, `gemini-3.1-pro-preview`, `gemini-omni-1.2-flash`, `gemini-omni-1.1-flash`, all image models (`gemini-nano-banana-2.1`, `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`, `gemini-2.5-flash-image`), and `gemini-2.5-pro-preview-tts`. Grounding with Google Search and Maps is "Not available" on the free tier. — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- The Free plan card lists "Limited access to certain models", "Free input & output tokens" and "Content used to improve our products". — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- "Rate limits are applied per project, not per API key. Requests per day (RPD) quotas reset at midnight Pacific time." Dimensions are RPM, TPM (input) and RPD, with TPD on some models. "Limits vary depending on the specific model." "Rate limits are more restricted for experimental and preview models." "Specified rate limits are not guaranteed and actual capacity may vary." The page gives numbers only for Omni Flash (free: "Not supported") and sends users to the logged-in AI Studio page (`https://aistudio.google.com/rate-limit`) for active limits. — [Gemini Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- Usage tiers: Free ("Active project or free trial; some projects need verification"). Tier 1 ("Set up and link an active billing account", cap $250). Tier 2 ("Paid $100 + 3 days", cap $2,000). Tier 3 ("Paid $1,000 + 30 days"). Spend-based rate limits (rolling 10 min) are N/A on Free and $10 on Tier 1. — [Gemini Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- Third-party (undated, retrieved 2026-11-06; says the figures come from "the Free Tier comparison in Google AI Studio Rate Limits"). Gemini Model | RPM | TPM | RPD: 3.8 Flash 5 | 250,000 | 20. 3.6 Flash 5 | 250,000 | 20. 3.5 Flash-Lite 10 | 250,000 | 500. 3.1 Flash-Lite 10 | 250,000 | 500. "The quotas apply per project and model." — [ScriptByAI, Gemini API Free Tier Limits](https://www.scriptbyai.com/gemini-api-free-tier-limits/)
- Corroboration of the 20 RPD figure: a forum post (2026-09-17) quotes the 429 `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaMetric: generativelanguage.googleapis.com/generate_content_free_tier_requests`, `quotaValue: 20` for `gemini-3.8-flash`, `gemini-3.7-flash` and `gemini-3.6-flash`. — [Google AI Developers Forum (2026-09-17)](https://discuss.ai.google.dev/t/gemini-api-free-tier-daily-quota-does-not-reset-persistent-429-resource-exhausted/183406)
- Model specs (each model page): `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` and `gemini-3-flash-preview` all have input token limit 1,048,576 and output token limit 65,536, and accept text, image, video, audio and PDF. — [gemini-3.8-flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [gemini-3.5-flash-lite model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- Recent changes, all from the release notes:
  - 2026-11-02: `gemini-2.5-pro` and `gemini-2.5-flash` deprecated. "Effective December 1, 2026, all API requests to `gemini-2.5-pro` will automatically route to `gemini-3.8-flash`", and `gemini-2.5-flash` will route to `gemini-3.5-flash-lite`, "billed at" the target's rates. Breaking changes: function calling and `thought_signature` validation, `thinking_level` replacing `thinking_budget`, removed sampling parameters and prefilled model turns.
  - 2026-10-08: `gemini-3.7-flash` is now routed to `gemini-3.8-flash`, and `gemini-3.5-flash` to `gemini-3.6-flash`.
  - Sampling parameters `temperature`, `top_p` and `top_k` are deprecated.
  - Gemini 3 Pro Preview was shut down (announced for March 9, 2026).
  - `gemini-3.1-flash-lite-preview` was deprecated in May 2026.

  — [Gemini API release notes](https://ai.google.dev/gemini-api/docs/changelog)

### Inferences
- Limits are per project and per model, so the SDK's "bench only that model" approach fits Gemini. Several API keys in one project share quota.
- The 20 RPD on the Flash models is very small. In rotation, treat the Flash-Lite models (about 500 RPD) as the volume workhorses. Once a daily `...PerDay...-FreeTier` 429 hits, bench that model until the next midnight Pacific.
- After 2026-12-01, hardcoded `gemini-2.5-*` IDs will silently hit another model and draw on *that* model's quota. Already-routed IDs (`gemini-3.7-flash`, `gemini-3.5-flash`) likely share their target's quota: the forum post shows a separate `quotaValue: 20` error for `gemini-3.7-flash`, but how routing interacts with quotas is unverified. Presets should use current IDs.

### Gaps
- No official current numbers exist for free-tier RPM, TPM or RPD per model. Google moved them behind the logged-in AI Studio page. The ScriptByAI numbers are third-party and undated. Only the 20 RPD figure is backed by a primary-ish observation (a user's 429 body).
- The free limits for `gemini-3-flash-preview`, `gemini-2.5-flash-lite` and Gemma 4, and Gemma 4's exact API model ID, were not found.

## Gemini API: 429 shape, other error codes, and headers

### Takeaway
A Gemini 429 is `{"error":{"code":429,"message":...,"status":"RESOURCE_EXHAUSTED","details":[...]}}`. When `details` is present it holds a `QuotaFailure` (with `quotaId`, which tells per-minute from per-day and free from paid) and a `RetryInfo` with `retryDelay` such as `"40s"`. The message may also say "Please retry in 40.15s". Bare 429s without details also occur. On the OpenAI-compat endpoint the body was seen **wrapped in a JSON array**. No `Retry-After` or `x-ratelimit-*` headers are documented.

### Cited Findings
- "If you receive an error indicating that you should retry your request (such as a `429 RESOURCE_EXHAUSTED` or `503 UNAVAILABLE`), we recommend implementing an exponential backoff strategy." "Do not retry on client errors (like `400`, `402`, or `403`) as they indicate issues like invalid API keys, depleted Prepay credits, or bad syntax." — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)
- Spend-based limit (paid tiers only): "If you hit a spend-based rate limit, the API returns a `429 RESOURCE_EXHAUSTED` error." — [Gemini Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- Detailed 429 (forum, 2026-02-06, `gemini-2.5-flash`, paid): `{"error":{"code":429,"message":"You exceeded your current quota ... * Quota exceeded for metric: ..., limit: 1000000, model: gemini-2.5-flash\nPlease retry in 40.152753394s.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"...Help","links":[...]},{"@type":"...QuotaFailure","violations":[{"quotaMetric":"...","quotaId":"GenerateContentPaidTierInputTokensPerModelPerMinute","quotaDimensions":{"model":"gemini-2.5-flash","location":"global"},"quotaValue":"1000000"}]},{"@type":"...RetryInfo","retryDelay":"40s"}]}}`. The forum software truncated the `@type` URLs. The same user also got the short form `{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`. — [Google AI Developers Forum (2026-02-06)](https://discuss.ai.google.dev/t/gemini-api-error-code-429-resource-exhausted-paid-plan/121679)
- Free-tier daily 429 identifiers (forum, 2026-09-17): `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaMetric: generativelanguage.googleapis.com/generate_content_free_tier_requests`. — [Google AI Developers Forum (2026-09-17)](https://discuss.ai.google.dev/t/gemini-api-free-tier-daily-quota-does-not-reset-persistent-429-resource-exhausted/183406)
- OpenAI-compat endpoint 429, as surfaced by the OpenAI Python client (forum, 2024-12-02): `Error code: 429 - [{'error': {'code': 429, 'message': 'Resource has been exhausted (e.g. check quota).', 'status': 'RESOURCE_EXHAUSTED'}}]`. Note the list wrapper. Other users saw frequent 429s on the compat endpoint with `stream=True` while native calls worked. — [Google AI Developers Forum (2024-12)](https://discuss.ai.google.dev/t/gemini-api-returns-429-issue-when-using-openai-compatible-api/52253)
- Spurious 429s (forum, 2026-08-06, `gemini-3.1-flash-lite`): about 40% of requests returned the short 429 while console quota use was 0.03%. "The OpenAI-compatibility endpoint reports this same condition as 400 INVALID_ARGUMENT." Google staff pointed to a report form; no root cause was posted. — [Google AI Developers Forum (2026-08)](https://discuss.ai.google.dev/t/429-resource-exhausted-at-0-03-of-quota-reproducible-across-projects-tiers-and-endpoints/177317)
- Verification and eligibility errors: an unverified free-tier project returns `403` (`FREE_TIER_VERIFICATION_REQUIRED`) with the message "Project 123456789012 must be verified before it can use the Gemini API Free tier. Verify the project at https://aistudio.google.com/verify?project=... or set up billing ...". — [Gemini Project and account verification](https://ai.google.dev/gemini-api/docs/verification). A restricted project returns `403 PERMISSION_DENIED` ("Your project has been denied access"), with statuses "Verification needed", "Restricted", "Free tier unavailable", "Payment needed" and "Unavailable". "Creating a new API key in the same project won't resolve the error." — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)
- Deprecation headers: "Gemini 4 and Gemini 3.6+ models ignore custom values for `temperature`, `top_p`, and `top_k` and return deprecation warning headers (and will return `400` errors in a future release)." — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)
- The "API errors" page codes (`rate_limit_exceeded` for per-minute and per-second limits, `quota_exceeded` for "daily quota", `too_many_requests`, `payment_required` 402, `service_unavailable` 503, `model_not_found` 404) are explicitly **Interactions API** codes in a `{"error":{"code":"snake_case","message":...}}` format. They are not the generateContent or OpenAI-compat format. — [Gemini API errors](https://ai.google.dev/gemini-api/docs/api-errors)

### Inferences
- Parser: if the body is a JSON list, take element 0. Read `error.status == "RESOURCE_EXHAUSTED"`. Scan `error.details[]` for an `@type` ending in `google.rpc.RetryInfo` and parse `retryDelay` (a protobuf duration string such as `"40s"` or `"0.5s"`). Fall back to the regex `retry in ([\d.]+)s` on `message`.
- Classify by `QuotaFailure.violations[].quotaId`: `PerDay` → bench the model until midnight America/Los_Angeles. `PerMinute` → bench for `retryDelay` or about 60 s. A missing `details` field → short backoff, since it may be a transient or capacity 429.
- 403 (verification, restricted, free tier unavailable) and 402 are **project-level** failures. Disable the whole Gemini provider rather than benching one model.
- Do not rely on a `Retry-After` header for Gemini. None is documented, and the 2026-08 forum report notes its absence. No `x-ratelimit-*` headers are documented, so proactive throttling must rely on client-side counters seeded with the limits.

### Gaps
- No primary Google doc gives the exact OpenAI-compat error envelope in 2026 (list-wrapped or not), whether `details` survives through the compat layer, or whether the compat layer still turns some 429s into 400 (single user report).
- Whether successful responses carry any rate-limit headers is unverified.

## Gemini API: OpenAI-compatibility quirks

### Takeaway
Supported: chat, streaming, tools with `tool_choice`, `response_format` structured output, image, audio and video input, embeddings, and `/models`. `reasoning_effort` maps onto Gemini thinking levels; it cannot be combined with `thinking_config`, and it cannot turn thinking off on Gemini 3.x. Gemini-only fields go in a body field literally named `extra_body` → `google`.

### Cited Findings
- `reasoning_effort` mapping (Gemini 3 Flash / 3.1 Flash-Lite): `minimal`→`minimal`, `low`→`low`, `medium`→`medium`, `high`→`high`. Gemini 2.5: `minimal`/`low`→1,024 thinking budget, `medium`→8,192, `high`→24,576. "you can set `reasoning_effort` to `"none"` for 2.5 models. Reasoning cannot be turned off for Gemini 2.5 Pro or 3 models." "`reasoning_effort` and `thinking_level`/`thinking_budget` overlap functionality, so they can't be used at the same time." — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- REST form of Gemini-specific fields: `"extra_body": {"google": {"thinking_config": {"thinking_level": "low", "include_thoughts": true}}}`. In the OpenAI Python SDK that becomes `extra_body={'extra_body': {...}}` because of double nesting. Other `extra_body` fields: `cached_content`, `thinking_config` (chat); `aspect_ratio`, `generation_config`, `safety_settings`, `tools` (images). "Gemini 3 supports OpenAI compatibility for thought signatures in chat completion APIs." — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- `service_tier` accepts `"priority"` or `"flex"`; the default `standard` equals OpenAI's `default`. Function calling is supported with `tool_choice="auto"` (doc example). Structured output is supported through `response_format` (Pydantic and Zod examples). For image generation, "Any other parameters not listed here ... will be silently ignored." — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- Parameter validity: candidate count is "unsupported on Gemini 3 and 4". `temperature` and `top_p` are "ignored on Gemini 3.6+ and 4; keep at default on earlier Gemini 3 models". — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)
- Thinking is on by default for Gemini 3.x and raises latency and token use. — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)
- The mapping table only has columns for Gemini 4 Argon, 3.1 Flash-Lite, 3 Flash and 2.5. None of the current free models `gemini-3.8-flash`, `gemini-3.6-flash` or `gemini-3.5-flash-lite` is listed. — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai). The `gemini-3.8-flash` model page says Thinking is "Supported (low, medium, high)" and "Note: `minimal` is not supported and returns an error." — [gemini-3.8-flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- The docs show `stream_options={'include_usage': True}` in a streaming example on `gemini-3.8-flash`, together with `extra_body.google.cached_content`. — [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)

### Inferences
- For Gemini, the SDK should drop `temperature`, `top_p`, `top_k` and `n>1`, and never send `reasoning_effort` together with `extra_body.google.thinking_config`.
- With thinking on by default and only 5 to 10 RPM on the free tier, latency per call is high. Free-tier TPM (input only) is unlikely to bind before RPD does.

### Gaps
- `reasoning_effort` → thinking-level mapping for `gemini-3.6-flash` and `gemini-3.5-flash-lite` is undocumented. `reasoning_effort="minimal"` probably errors on `gemini-3.8-flash` (per the model page note), but this is unverified through the compat layer.
- Not verified: the exact chunk carrying usage when `stream_options.include_usage` is set (documented only by example), `tool_choice: "required"` and named-function support, whether thought summaries appear as `reasoning_content` or another field, and the exact compat behavior for unsupported OpenAI params (silent ignore or 400) for chat.

## Gemini API: ToS, data use, regions (EU/UK/Korea), verification, and multiple projects

### Takeaway
On the free tier ("Unpaid Services"), Google uses prompts and responses to improve products, and human reviewers may read them. South Korea, the UK and EU countries are on the available-regions list. The EEA/CH/UK clause limits **serving end users there** to Paid Services; it does not bar developers located there. Developers must be 18+ and the service is for professional or business use. Extra free projects need mobile-device verification.

### Cited Findings
- "When you use Unpaid Services, including, for example, Google AI Studio and the unpaid quota on Gemini API, Google uses the content you submit to the Services and any generated responses to provide, improve, and develop Google products and services and machine learning technologies ... human reviewers may read, annotate, and process your API input and output ... **Do not submit sensitive, confidential, or personal information to the Unpaid Services.**" — [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms)
- "You may only access the Services (or make API Clients available to users) within an available region. **You may use only Paid Services when making API Clients available to users in the European Economic Area, Switzerland, or the United Kingdom.**" Use is "for developers building with Google AI models for professional or business purposes, not for consumer use". "You must be 18 years of age or older." No building competing models. — [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms)
- The available-regions list includes "South Korea", "United Kingdom", "Germany", "France" and others. For Colab, region checks apply to the Colab instance location. — [Gemini Available regions](https://ai.google.dev/gemini-api/docs/available-regions)
- "Google AI Studio and the Gemini API require mobile device verification on some free tier projects and accounts. ... Project verification: Required for the free tier on additional projects or projects created outside Google AI Studio." It uses a QR code and reCAPTCHA on a physical phone, without sharing a phone number. Rooted phones fail. "Creating a new API key in the same unverified project does not bypass verification." — [Gemini Project and account verification](https://ai.google.dev/gemini-api/docs/verification)
- "Not all Google Cloud projects or accounts are eligible for the Gemini API free tier ... Upgrading a project to the paid tier doesn't unlock free-tier quota." — [Gemini Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting)

### Inferences
- A free-tier SDK whose output reaches end users in the EEA, Switzerland or the UK breaches the Gemini terms. That is a documentation and README concern for `freeai`.
- Multi-project rotation is technically gated by per-project quotas plus device verification for extra free projects. That is Google's anti-multiplication mechanism.

### Gaps
- The Additional Terms text I scanned has no explicit "multiple accounts" or "circumvent limits" clause. The base Google APIs Terms of Service (https://developers.google.com/terms) were not fetched, so a clause against circumventing API limits is unverified.

---

## Mistral (La Plateforme): base URL, `/models`, and model ID format

### Takeaway
`https://api.mistral.ai/v1` answers on `/models` and `/chat/completions` (a live probe got 401 for the bad key). IDs are dated API names (`mistral-small-2603`, `ministral-8b-2512`). The live 401 body is `{"detail": "..."}`, not the documented error object.

### Cited Findings
- Live probe 2026-11-06, `GET https://api.mistral.ai/v1/models` and `POST .../v1/chat/completions` with a bad key: `HTTP/2 401`, body `{"detail":"Invalid API Key"}`, headers `mistral-correlation-id`, `x-kong-request-id` and `x-kong-response-latency`. — live probe of [api.mistral.ai/v1/models](https://api.mistral.ai/v1/models)
- Current GA or preview text models (Model | API name | Context):
  - Mistral Small 4: `mistral-small-2603`, 256k
  - Mistral Large 4: `mistral-large-4` (Public Preview), 1M
  - Ministral 3 14B / 8B / 3B: `ministral-14b-2512` / `ministral-8b-2512` / `ministral-3b-2512`, 256k
  - Codestral: `codestral-2508`, 128k
  - Third-party models: `zai-glm-5-3-flash` (1M), `zai-glm-5-3` (1M), `deepseek-v4-1-flash-1017` (Public Preview, 1M)
  - Deprecated: `mistral-medium-3-5` (retires 2027-04-30), `mistral-large-2512` (retires 2027-04-30), `magistral-*` (retired, or deprecated with retirement passed), `mistral-small-2506` and `open-mistral-nemo-2407` (status "Deprecated", but the listed retirement date 2026-07-31 has passed)

  — [Mistral Models Overview](https://docs.mistral.ai/models)
- ID inconsistency: the reasoning page calls Mistral Large 4 `mistral-large-4-0`, while the models page says `mistral-large-4`. — [Mistral Reasoning](https://docs.mistral.ai/inference/conversations/reasoning) vs [Mistral Models Overview](https://docs.mistral.ai/models)

### Inferences
- Error parsing must handle the gateway form `{"detail": ...}` (seen at least for 401) as well as the documented `{"object":"error",...}`.

### Gaps
- Not verified: the `/v1/models` response shape and whether `-latest` aliases (for example `mistral-small-latest`) still resolve. They are not on the current models page.

## Mistral: free API tier status, limits, scope, and recent changes

### Takeaway
**Mistral's official docs (2026-11-06) say the free plan ("Free mode") does not include API access.** Creating API keys needs billing activation and prepaid pay-as-you-go credits. The "Experiment" help article now returns 404. The free API tier launched on 2024-09-17. No discontinuation announcement was found. Where API access exists, limits are per model, as TPM plus **requests per second**.

### Cited Findings
- "**Free mode** includes a monthly usage budget for trying Vibe and Studio. It doesn't include API access. Creating API keys and using the API require billing activation and pay-as-you-go." — [Mistral Usage and limits](https://docs.mistral.ai/admin/billing-usage/usage-limits)
- "Available plans include Free mode, Pro, Education, Team, and Enterprise. Free mode is the default state for new accounts." "Each Mistral plan includes monthly usage that is shared across Studio, the API, and Vibe Code." "Pay-as-you-go is prepaid, so your Organization needs credits before you enable it." — [Mistral Subscription](https://docs.mistral.ai/admin/billing-usage/subscriptions)
- "Completion rate limits are listed per model and include: Tokens per minute ... Requests per second." Limits are shown in Admin Panel → API → Limits (`https://admin.mistral.ai/plateforme/limits`). Org and Workspace monthly spending limits exist; "If the Organization reaches its monthly spending limit, API access can be suspended." — [Mistral Usage and limits](https://docs.mistral.ai/admin/billing-usage/usage-limits)
- Changelog, 2024-09-17: "We introduced a free API tier on La Plateforme." — [Mistral Changelog](https://docs.mistral.ai/resources/changelogs)
- Fetching `https://help.mistral.ai/en/articles/455206-how-can-i-try-the-api-for-free-with-the-experiment-plan` on 2026-11-06 returned the Help Center's "404 Uh oh. That page doesn't exist." — [Mistral Help Center article (now 404)](https://help.mistral.ai/en/articles/455206-how-can-i-try-the-api-for-free-with-the-experiment-plan)
- The pricing page FAQ says "Mistral's Free plan provides limited access to Vibe ... On the Free plan, you also get access to Mistral Studio". The plan table shows "Model training: Opt-out". A keyword scan of the page found no "Experiment" plan. — [Mistral pricing](https://mistral.ai/pricing)
- **Conflict:** a web-search summary (2026) still described the Experiment plan as available: phone verification, no card, "requests ... may be used to train Mistral's models". It cited a help-center snapshot and third-party guides from January and May 2026, including a January 2026 staff post saying free usage would continue under the Experiment plan. These could not be re-checked; the help article now 404s. — [unified.to guide](https://unified.to/blog/how_to_get_a_mistral_ai_api_key_and_connect_it_to_your_product), [fremodel guide (2026)](https://fremodel.pages.dev/providers/mistral-ai)

### Inferences
- Treat Mistral as **not free for API use as of 2026-11-06**. The free Experiment plan appears discontinued sometime after early to mid 2026. Keys created earlier under that plan may or may not still work; that is unverified. A Mistral preset should be opt-in and labeled paid or credits-required.
- If a legacy key works, expect 402 when included usage or credits run out. That is account-level.

### Gaps
- No official announcement or date of the Experiment-plan shutdown was found.
- Whether existing Experiment-plan keys still work is unknown.
- The last published Experiment-plan limits (historically described as about 1 RPS) could not be confirmed from a primary source.

## Mistral: 429 shape, other error codes, and rate-limit headers

### Takeaway
The docs list 429 causes as requests per second, tokens per minute **or concurrent requests**, and say to check `Retry-After`. The documented error body is `{"object":"error","message","type","param","code"}`, and `rate_limit_error` is one of the listed `type` values. Presumably 429s use it, but no 429 example is documented. 402 marks billing exhaustion. Rate-limit headers on success are not documented.

### Cited Findings
- 429: "You have exceeded the rate limit for your subscription tier. Common causes: Too many requests per second; Token-per-minute quota exceeded; Concurrent request limit reached. Resolution: Implement exponential backoff. Check the `Retry-After` response header." — [Mistral Error glossary](https://docs.mistral.ai/resources/error-glossary)
- 402: "The prepaid credit balance has reached zero. The Organization or Workspace monthly spending limit is reached. Included monthly usage is exhausted and pay-as-you-go isn't enabled. A recent invoice payment failed." 403 can mean "Accessing a model not available on your subscription tier". 400 can mean "`temperature` outside 0 to 1 range". 422 is a validation error. 500, 502, 503 and 504 are transient. — [Mistral Error glossary](https://docs.mistral.ai/resources/error-glossary)
- Format: `{"object": "error", "message": "...", "type": "invalid_request_error", "param": "model", "code": "unknown_model"}`. `type` ∈ {`invalid_request_error`, `authentication_error`, `rate_limit_error`, `server_error`}. — [Mistral Error glossary](https://docs.mistral.ai/resources/error-glossary)
- Contradicting the documented format, the live 401 body was `{"detail":"Invalid API Key"}`. — live probe 2026-11-06 of [api.mistral.ai/v1/chat/completions](https://api.mistral.ai/v1/chat/completions)

### Inferences
- A 429 caused by the concurrency limit should get a short retry, not a long bench. Without a body field telling the causes apart, honor `Retry-After` when present and otherwise use a short exponential backoff.
- 402 and 401 are account-level: disable the provider.

### Gaps
- No primary example of a Mistral 429 body or its `code` values was found.
- The names of rate-limit headers on success responses (if any) are undocumented on the pages fetched.

## Mistral: OpenAI-compatibility quirks

### Takeaway
Chat is OpenAI-like, with Mistral additions such as `safe_prompt`. Reasoning models with `reasoning_effort="high"` return **`message.content` as a list of chunks** (`thinking` plus `text`) instead of a string, and `zai-glm-5-3` always does. Naive `content: str` parsers will break.

### Cited Findings
- Reasoning: `mistral-small-2603` and `mistral-medium-3-5` accept `reasoning_effort` (`high` or `none`). `mistral-large-4(-0)` accepts `low`, `medium` and `high`. `zai-glm-5-3` and `zai-glm-5-3-flash` accept `low`, `high` and `max`, and "`none` is not supported". — [Mistral Reasoning](https://docs.mistral.ai/inference/conversations/reasoning)
- "When `reasoning_effort` is set to `"high"`, the response `message.content` is a **list of chunks** instead of a plain string. ... `ThinkChunk` (`type: "thinking"`): ... The `thinking` field is itself a list of `TextChunk` objects. `TextChunk` (`type: "text"`): contains the final answer. When `reasoning_effort` is `"none"`, `message.content` is a plain `str`." For `zai-glm-5-3`, "`message.content` is always returned as a list of chunks ... there is no plain-string mode." — [Mistral Reasoning](https://docs.mistral.ai/inference/conversations/reasoning)
- `safe_prompt` (boolean) adds a moderation system prompt. — [Mistral Chat completions](https://docs.mistral.ai/inference/conversations/chat-completion)
- Structured outputs: JSON mode via `response_format: {type: "json_object"}` — [Mistral JSON Mode](https://docs.mistral.ai/inference/conversations/structured-output/json_mode); custom JSON-schema structured outputs — [Mistral Custom Structured Outputs](https://docs.mistral.ai/inference/conversations/structured-output/custom); function calling — [Mistral Function Calling](https://docs.mistral.ai/inference/conversations/function-calling).
- `stream_options` and `include_usage` do not appear in the Mistral chat-completions guide fetched 2026-11-06 (grep found no hits). — [Mistral Chat completions](https://docs.mistral.ai/inference/conversations/chat-completion)

### Inferences
- The SDK's response normalizer must flatten list-typed `content`: join the `text` chunks, and send the `thinking` chunks to a reasoning field.

### Gaps
- Not verified: whether `stream_options.include_usage` is accepted (undocumented in the guide), `tool_choice` values (`any` vs OpenAI `required`), `parallel_tool_calls`, and which OpenAI params cause 400 or 422 (for example `n`, `logit_bias`, `seed` vs `random_seed`).
- Max output tokens per Mistral model were not captured. Only context lengths were.

---

## Cross-platform: how should `freeai` classify 429 and other errors for these four presets?

### Takeaway
All four set limits per model inside an org or project, so model-level benching on 429 is broadly correct. Each provider, though, has **account-level** failures (402, 401, some 403s) that must disable the whole provider. The retry signal differs per provider:
- Groq: `retry-after` plus `x-ratelimit-reset-*`
- Gemini: `RetryInfo.retryDelay` in the body
- Mistral: `Retry-After`
- Cerebras: unknown

### Cited Findings
- Groq limits are org-level per model; `retry-after` is seconds and only on 429; the `x-ratelimit-*-requests` headers count RPD and the `*-tokens` headers count TPM. — [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- Cerebras limits are org-level per model, plus project-level limits, using a token bucket; it returns 402 PaymentRequired. — [Cerebras Rate Limits](https://inference-docs.cerebras.ai/support/rate-limits), [Cerebras Error Codes](https://inference-docs.cerebras.ai/support/error)
- Gemini limits are per project and per model; RPD resets at midnight Pacific; 403 `FREE_TIER_VERIFICATION_REQUIRED` and 403 `PERMISSION_DENIED` are project-level. — [Gemini Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [Gemini Project and account verification](https://ai.google.dev/gemini-api/docs/verification)
- Mistral limits are per model (TPM and RPS); 429 can come from the concurrency limit; 402 covers billing exhaustion. — [Mistral Usage and limits](https://docs.mistral.ai/admin/billing-usage/usage-limits), [Mistral Error glossary](https://docs.mistral.ai/resources/error-glossary)

### Inferences
- Suggested classification table (inference from the findings above):

| Signal | Scope | Action |
|---|---|---|
| Groq 429, TPM/RPM | model | bench `retry-after` s (or the parsed message duration) |
| Groq 429, RPD/TPD, or `x-ratelimit-remaining-requests: 0` | model | bench until `x-ratelimit-reset-requests` |
| Gemini 429 with `quotaId` containing `PerDay` | model | bench until next 00:00 America/Los_Angeles |
| Gemini 429 with `PerMinute` or a `retryDelay` | model | bench `retryDelay` |
| Gemini 429 without `details` | model | short backoff (may be transient or spurious) |
| Gemini 503 `UNAVAILABLE` | model | short backoff and rotate |
| Gemini 403, 402 | provider | disable Gemini |
| Cerebras 429 | model | short bench (continuous refill); longer if the message names hourly or daily caps (Free Trial) |
| Cerebras 402 | provider | trial or credits exhausted; disable |
| Mistral 429 | model | `Retry-After` or a short backoff |
| Mistral 402, 401 | provider | disable |
| 400 / 422 for unsupported params (Groq `logprobs`, Cerebras `tools`+`response_format` on gpt-oss-120b, Gemini `n>1`) | model capability | do not count as a rate limit; skip that model for this request shape |

- Header parsing: none of the four documents a bare `x-ratelimit-reset` header. `freeai`'s parser for it is harmless but probably unused for these presets. The Groq duration format (`2m59.56s`) is confirmed. Gemini needs body parsing for `retryDelay`.
- Body parsing must accept all of these:
  - `{"error":{...}}` (Groq; Gemini native)
  - `[{"error":{...}}]` (Gemini compat, observed 2024-12)
  - a flat `{"message","type","param","code"}` (Cerebras)
  - `{"object":"error",...}` (Mistral docs)
  - `{"detail": "..."}` (Mistral gateway)

### Gaps
- None of the four providers' real 429 responses could be observed in this session (no keys). Live authenticated captures of a 429 on each provider remain the top verification item before the classifier ships.
