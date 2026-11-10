# OpenRouter Free-Model Tier: Limits, Errors, Routing, Signals, Privacy, Churn (for the `freeai` rotation SDK)

Research date: 2026-11-06. "Official" = openrouter.ai docs, pricing page, OpenAPI spec, or the live public API. "Community" = third-party blogs, forums, or GitHub issues, each dated. Live API numbers come from unauthenticated calls I made to `https://openrouter.ai/api/v1/...` on 2026-11-06.

## Q1. Scope of free limits: is the 20 RPM per account or per model, is the daily cap shared, and do failed/429 requests count?

### Takeaway
The daily cap is officially account-wide, shared by all `:free` models, and resets at 00:00 UTC. Failed attempts officially count toward it. No official source says whether the 20 RPM cap is per account or per model. A 2025 log shows a separate per-model/per-endpoint limit (`limit_rpm/<model>/...`) on top of the account cap, and the limits page still says "different rate limits for different models". Treat the RPM limit as possibly two-layered: an account bucket plus per-model buckets.

### Cited Findings
- Free-variant limits (official, current): fewer than 10 credits purchased all-time gives 20 RPM and 50 RPD. At least 10 credits gives 20 RPM and 1000 RPD. These apply to "a free model variant (with an ID ending in `:free`)". — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- The `free_model_daily_requests` object is the "Free-model (`:free` variant) daily request quota **for the account that owns the key**", and "the counter resets at UTC midnight". — [Get current API key (OpenAPI)](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key)
- The FAQ says free models have "low rate limits (**50 requests per day total**)". — [FAQ](https://openrouter.ai/docs/faq)
- The pricing FAQ says free models are "capped at 20 requests per minute and 50 per day, rising to 1,000 per day once your account has bought at least $10 of credits. **Failed attempts count toward that daily cap**, and a negative balance blocks free models until you add credits." — [Pricing](https://openrouter.ai/pricing)
- The limits page opens with: "Making additional accounts or API keys will not affect your rate limits, as we govern capacity globally. We do however have different rate limits for different models, so you can share the load that way if you do run into issues." — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- `free_model_daily_requests` reports only the daily counter. "The per-minute limit is not reported there." — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- Community, 2025-03: a free Gemini model returned `"Rate limit exceeded: limit_rpm/google/gemini-2.0-flash-thinking-exp-01-21/54276bca-..."` with `X-RateLimit-Limit: "4"`, and `"limit_rpd/google/gemini-2.0-flash-thinking-exp-01-21/..."` with `X-RateLimit-Limit: "80"`. These limits were specific to the model and endpoint, and sat below the account-level 20/200 limits of that time. — [LiteLLM #9035 (2025-03-06)](https://github.com/BerriAI/litellm/issues/9035)
- Community, 2026-10: TrueFoundry describes the cap as "20 requests a minute on any free model" and says "Failed requests can still count against the daily allowance". — [TrueFoundry, free models (2026-10-08)](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you); [TrueFoundry, rate limits (2026-10-05)](https://www.truefoundry.com/blog/openrouter-rate-limits-your-free-tier-depends-on-what-youve-already-paid)
- Community, contradicting: CostGoat says "Limits apply per-model", but it also quotes the old 200 RPD figure, so its numbers are stale. — [CostGoat (updated 2026-11-04)](https://costgoat.com/pricing/openrouter-free-models)
- Community, 2025-04: Janitor AI says that for `free-models-per-day`, "Even error messages count toward this limit". — [Janitor AI help (2025-04-08)](https://help.janitorai.com/en/article/openrouter-error-guide-10ear52/)
- A negative credit balance can cause 402 errors "including for free models". — [Limits docs](https://openrouter.ai/docs/api/reference/limits)

### Inferences
- Rotating across `:free` models does not raise the daily budget. 50 or 1000 is one shared pool, so the SDK should keep one account-wide daily counter and seed it from `GET /api/v1/key`.
- Every 429 retry spends daily budget. That includes upstream "temporarily rate-limited upstream" 429s, which the pricing page's "failed attempts" wording covers. Retries on a 50-RPD account must be capped hard, and immediate retries on the same model are expensive.
- Keep the client-side 20/min throttle global across models. Also keep per-model cooldowns, because per-model `limit_rpm/...` buckets have been observed (2025) and the docs still mention per-model limits.

### Gaps
- No official statement says whether the 20 RPM bucket is per account or per model. A test would settle it: send about 25 requests in 60 s, alternating two `:free` models, and read `X-RateLimit-Limit` and the message on the first 429.
- Not confirmed: whether a request rejected by OpenRouter's own rate limiter counts against the daily cap, as opposed to provider failures. "Failed attempts count" is not broken down further.
- Not confirmed: whether the 2025 `limit_rpm/`/`limit_rpd/` per-model limits still exist in 2026. No 2026 sighting found.

## Q2. Exact 429 formats: `X-RateLimit-*` units, platform vs upstream message wording, and where the data actually appears

### Takeaway
`X-RateLimit-Reset` is a Unix epoch timestamp in milliseconds. For the daily cap it equals the next 00:00 UTC. In every observed platform 429 body, the `X-RateLimit-*` values sit inside JSON at `error.metadata.headers`. The platform body can also carry `provider_name: null`, so the key is present with a null value. Upstream 429s carry `metadata.provider_name` (non-null) and `metadata.raw` containing "is temporarily rate-limited upstream". Free-model 429s have also arrived as HTTP 200 with an `error` body.

### Cited Findings
- Official: platform 429s for free-model or key-management limits carry `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`. Successful inference responses carry no `X-RateLimit-*` headers. "When every attempted provider returned a retry hint, the error response also carries a `Retry-After` header." — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- Official: for the key-management limit, "The `X-RateLimit-Reset` header gives the reset time in milliseconds since the Unix epoch." — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- Official: on 429 and 503, OpenRouter "may include a standard HTTP `Retry-After` response header indicating how many seconds to wait". — [Errors docs](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- Official generic 429 body: `{"error":{"code":429,"message":"Rate limit exceeded","metadata":{"error_type":"rate_limit_exceeded"}}}`. Provider-originated errors add `metadata.provider_code`, for example `"rate_limited"`. — [Errors docs](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- Official typed codes: `rate_limit_exceeded` (429), `provider_overloaded` (503), `provider_unavailable` (502), `timeout` (504), `server` (500, message masked), `unmapped` (500). For Chat Completions they appear at `error.metadata.error_type`. — [Errors docs](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- Community, 2025-12-24, platform daily-cap body (verbatim): `{'error': {'message': 'Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1766620800000'}, 'provider_name': None}}}`. I computed 1766620800000 ms = 2025-12-25T00:00:00Z, the next UTC midnight. — [pyVideoTrans forum #2266](https://bbs.pyvideotrans.com/show/2266)
- Community, 2025-03, per-model body: `"Rate limit exceeded: limit_rpm/<model-permaslug>/<uuid>"` with `metadata.headers` `{X-RateLimit-Limit:"4", X-RateLimit-Remaining:"0", X-RateLimit-Reset:"1741294740000"}`, an epoch-ms timestamp at a minute boundary. The `limit_rpd/...` reset was `1741305600000` (UTC midnight). LiteLLM got this body with HTTP 200 and turned it into an empty `finish_reason='stop'` reply. — [LiteLLM #9035](https://github.com/BerriAI/litellm/issues/9035)
- Community, 2025-01: smolagents logs show `"HTTP/1.1 200 OK"` followed by `RateLimitError` on a `:free` model, which is the 429-in-200-body case again. — [smolagents #361](https://github.com/huggingface/smolagents/issues/361)
- Community, 2026-01, upstream 429 body: `{"code":429,"message":"Provider returned error","metadata":{"is_byok":false,"provider_name":"OpenInference","raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations"}}`. It happened on the first request of the day. — [LibreChat discussion #11480 (2026-01-22)](https://github.com/danny-avila/LibreChat/discussions/11480)
- Community message catalog (2025-04): "Rate limit exceeded: free-models-per-day" (429), "[model] is temporarily rate-limited upstream…" (429), "No endpoints found for model name" (404), "No allowed providers are available for the selected model" (404), "No endpoints found matching your data policy." (404), "Provider returned error" (502). — [Janitor AI help](https://help.janitorai.com/en/article/openrouter-error-guide-10ear52/)
- Official: rate-limit failures do not carry `openrouter_metadata`, even when the request opts in. — [Router Metadata docs](https://openrouter.ai/docs/guides/features/router-metadata)

### Inferences
- Classifier fixes for `freeai`:
  - Treat as model-level only when `error.metadata.provider_name` is non-null, or `metadata.raw` contains "rate-limited upstream".
  - Treat as account/platform-level when the message starts with `Rate limit exceeded: free-models-per-` (day observed; min is assumed).
  - Look for `X-RateLimit-*` in `error.metadata.headers` as well as in HTTP headers.
  - Treat `Rate limit exceeded: limit_rpm/` or `limit_rpd/` followed by a model slug as model-scoped and cool that model only.
- Compute the cooldown as `reset_ms/1000 - now`. For `free-models-per-day`, stop all `:free` traffic until that time; no amount of rotation helps. Prefer `Retry-After` (seconds) when present.
- Check the body for `error` on every 2xx response, and treat an empty completion with no usage as suspect.

### Gaps
- The string `free-models-per-min` appeared in no source. The per-minute platform message is unverified, so do not hard-code it as fact. Match a `free-models-per-` prefix and record real samples.
- Not confirmed: whether 2026 platform 429s also set real HTTP `X-RateLimit-*` headers. The docs say "headers", but every observed sample shows them in the body.
- No official retry window exists for upstream throttling.

## Q3. How `openrouter/free` works, and other routers or meta-models

### Takeaway
`openrouter/free` picks a free model at random from the pool, filtered by the capabilities your request needs. On its own it "falls back only to other free models", so it retries server-side within the free pool and never bills. You cannot steer it, and responses report the chosen model in `model`. It is a good last resort, but consecutive turns can land on different models with different data policies.

### Cited Findings
- How it works (official): (1) request analysis (vision, tools, structured outputs) → (2) filter free models → (3) **random selection** → (4) forward → (5) the response's `model` field shows the model used. "You cannot control which specific model is selected." — [Free Models Router docs](https://openrouter.ai/docs/guides/routing/routers/free-router)
- "On its own, the Free Models Router falls back only to other free models. If you add paid models to a `models` fallback array alongside `openrouter/free` or a `:free` model, OpenRouter uses those paid models… and bills them at their own price." — [Free Models Router cookbook](https://openrouter.ai/docs/cookbook/get-started/free-models-router-playground); same warning in [Free variant docs](https://openrouter.ai/docs/guides/routing/model-variants/free)
- Live catalog entry: `openrouter/free` was created 2026-02-01, has pricing `{"prompt":"0","completion":"0"}`, `context_length` 200000, input modalities `text, image`, and `supported_parameters` that include `tools`, `tool_choice`, `response_format`, `structured_outputs`, `reasoning`. Its `/endpoints` returns `"endpoints": []`. — [Live models API](https://openrouter.ai/api/v1/models); [Live endpoints](https://openrouter.ai/api/v1/models/openrouter/free/endpoints)
- With `X-OpenRouter-Metadata: enabled`, responses include `openrouter_metadata.strategy`, whose values include `free`, plus `attempt` (above 1 means fallback happened) and an optional `attempts[]` with per-attempt provider, model and status. — [Router Metadata docs](https://openrouter.ai/docs/guides/features/router-metadata)
- Community (2026-10): "two requests in the same conversation can be answered by different models from different companies under different data policies." — [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- Other router strategies listed in the metadata docs are `auto`, `latest`, `alias`, `fallback`, `pareto`, `bodybuilder` and `fusion`. The documented routers are Auto, Body Builder, Free, Latest Model Resolution, Pareto, Fusion, Jev and Switchyard. None of the others is documented as free-only. — [Router Metadata docs](https://openrouter.ai/docs/guides/features/router-metadata); [docs index](https://openrouter.ai/docs/llms.txt)
- Auto and Jev routers accept `allowed_models` or `excluded_models` wildcard filters, matched against the router pool. The Free router page documents no such filter. — [Auto Router docs](https://openrouter.ai/docs/guides/routing/routers/auto-router); [Free Models Router docs](https://openrouter.ai/docs/guides/routing/routers/free-router)

### Inferences
- With `openrouter/free`, server-side fallback can make several upstream attempts inside one client request. How those attempts count against the daily cap is undocumented, and "failed attempts count" suggests they might each count. Use it as the last resort, as the SDK already does, not as the primary rotation.
- Use `response.model` to attribute success or failure to the concrete free model, and feed that into the SDK's per-model health stats.
- A `models` array that lists only `:free` IDs gives server-side rotation without billing risk. The docs explicitly endorse this pattern.

### Gaps
- Not confirmed: whether one `openrouter/free` call counts as one request or as one per internal attempt against `free_model_daily_requests`.
- Not confirmed: whether the random pool includes zero-priced non-`:free` models such as `stealth/crouton-alpha`.
- No limit on the length of the chat `models` array was found in the spec. The Anthropic-skin `fallbacks` is capped at 3 entries.

## Q4. Which provider routing parameters keep requests free and reliable (`order`, `allow_fallbacks`, `sort`, `only`, `ignore`, `data_collection`, `zdr`, `require_parameters`, `max_price`)?

### Takeaway
In the live catalog every `:free` model has exactly one endpoint, so provider-level knobs (`order`, `only`, `ignore`, `sort`, `allow_fallbacks`) have nothing to choose between. The useful ones are a zero `max_price` as a billing guard, `require_parameters: true` for fail-fast capability matching, and `data_collection`/`zdr` used with care, because they can eliminate almost every free model.

### Cited Findings
- The `provider` object fields: `order`, `allow_fallbacks` (default true), `require_parameters` (default false), `data_collection` (`allow`|`deny`, default `allow`), `zdr`, `enforce_distillable_text`, `only`, `ignore`, `quantizations`, `sort` (`price`|`throughput`|`latency`, or `{by, partition}`), `preferred_min_throughput`, `preferred_max_latency` (seconds), `max_price` (`{prompt, completion, request, image}` in $/M tokens). — [Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection)
- `require_parameters: true` means "the request won't even be routed to that provider". Even without it, `tools`, `response_format` (including structured outputs) and `verbosity` act as a soft preference. But "If none of a model's providers support the parameter, the request is still routed to that model and the parameter is ignored." — [Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection)
- When `tools`/`tool_choice` are sent, routing makes a "best effort" to pick tool-capable providers. `max_tokens` restricts routing to providers that support that length. — [Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection)
- Default routing weighs effective price, throughput, TTFT, 429 rate, reliability and tool-call quality. "A provider whose errors or 429s are far worse… gets no traffic until it recovers." — [Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection)
- `models` fallbacks trigger on "Context length validation errors, Moderation flags, Rate-limiting, Downtime". Billing follows the model actually used. — [Model Fallbacks docs](https://openrouter.ai/docs/guides/routing/model-fallbacks)
- Live data (2026-11-06): `google/gemma-4-31b-it:free` has 1 endpoint (Google AI Studio), `nvidia/nemotron-3-ultra-550b-a55b:free` has 1 (Nvidia), `recursive/flint-v1.0:free` has 1 (Recursive), `stealth/crouton-alpha` has 1 (Stealth). — [Live endpoints, e.g. gemma](https://openrouter.ai/api/v1/models/google/gemma-4-31b-it:free/endpoints)
- Community (2026-10): "Every one of the 19 free models has exactly one endpoint… when it's busy or down there's nowhere to fall back to." — [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- The free endpoint can differ from paid hosts. Gemma 4 31B free caps output at 32,768 tokens, against up to 262,141 on paid hosts. — [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- The pricing table lists "Auto-routing, preferred vendor selections" and "Data Policy-Based Routing" as **No** for the Free plan and Yes for Standard. — [Pricing](https://openrouter.ai/pricing)

### Inferences
- Send `provider: {"max_price": {"prompt": 0, "completion": 0, "request": 0}}` with every request that uses a `models` array, as defense in depth against billing a paid fallback. Expect a 404/503 instead of a charge (untested).
- `require_parameters: true` is useful when the SDK sends `tools` or `response_format`. A free model whose single endpoint lacks the parameter is skipped, or fails fast inside a `models` array. Without it, the soft preference cannot remove the model and the parameter is silently dropped, which is the worst outcome for tool or JSON use. Also pre-filter locally on `supported_parameters` from the models/endpoints APIs.
- `order`, `only`, `ignore`, `sort` and `allow_fallbacks` add nothing for `:free` IDs today (one endpoint each). They only matter if a free model gains a second host.
- Free-plan accounts (never purchased credits) may not get data-policy routing or vendor preferences, per the pricing table. Whether that is enforced at request time is unknown.

### Gaps
- Not tested: whether `max_price` zero actually blocks paid fallbacks in a `models` array.
- Not known: how Free-plan restrictions on "preferred vendor selections" and "Data Policy-Based Routing" are enforced at request time (ignored vs rejected).

## Q5. Health and quality signals: Endpoints API fields, models-list quality filters, Generation API

### Takeaway
The Endpoints API gives per-endpoint `uptime_last_5m/30m/1d`, `status`, `supported_parameters` and, with an API key only, `latency_last_30m`/`throughput_last_30m` percentiles. Uptime **excludes rate-limited failures**, so it does not predict 429s. The models list supports server-side filters for Artificial Analysis indexes and `min_tool_success_rate`, and returns `benchmarks.artificial_analysis.{intelligence_index, coding_index, agentic_index}`. There is no tool-success-rate response field. `GET /api/v1/generation?id=` returns `provider_responses[]` covering every fallback attempt.

### Cited Findings
- Endpoints path: `GET /api/v1/models/{author}/{slug}/endpoints`. A `:free` slug works directly, e.g. `/models/google/gemma-4-31b-it:free/endpoints`. — [Endpoints API reference](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model); [Live](https://openrouter.ai/api/v1/models/google/gemma-4-31b-it:free/endpoints)
- `PublicEndpoint` fields: `name`, `model_id`, `model_name`, `context_length`, `pricing`, `provider_name`, `tag`, `quantization`, `max_completion_tokens`, `max_prompt_tokens`, `supported_parameters`, `supports_tool_choice{auto,none,required,function}`, `status` (enum 0, -1, -2, -3, -5, -10), `uptime_last_5m`, `uptime_last_30m`, `uptime_last_1d`, `latency_last_30m`, `throughput_last_30m`, `perf_last_30m_by_workload.text_generation{latency,throughput,request_count}`, `supports_implicit_caching`, `service_tier`, `native_tools`. — [Endpoints API reference](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)
- Uptime definition: "the share of minutes in which at least 80% of provider attempts succeeded, counting only minutes with 10 or more attempts. **Rate-limited and caller-caused failures are excluded.** Null when no minute had enough traffic." — [Endpoints API reference](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)
- Latency is TTFT. Latency and throughput percentiles are "Only visible when authenticated with an API key or cookie; returns null for unauthenticated requests". Throughput pN means the speed N% of requests reached, so p90 is below p50. — [Endpoints API reference](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)
- Live (unauthenticated, 2026-11-06): Gemma 4 31B free has `uptime_last_1d` 99.44 with 5m/30m null. Nemotron 3 Ultra free has 1d 89.24 and 30m 100. Flint free has 1d 94.10. All have `latency_last_30m` and `throughput_last_30m` null. — [Live endpoints](https://openrouter.ai/api/v1/models/nvidia/nemotron-3-ultra-550b-a55b:free/endpoints)
- Models-list query params include `supported_parameters` (comma-separated), `output_modalities`, `input_modalities`, `endpoints`, `context`, `min_price`/`max_price` (prompt $/M), `min_output_price`/`max_output_price`, `min_age_days`/`max_age_days`, `min|max_intelligence_index`, `min|max_coding_index`, `min|max_agentic_index`, `min_tool_success_rate`/`max_tool_success_rate`, `zdr=true`, `distillable`, `region`, `q`, and `sort` (including `intelligence-high-to-low`, `coding-high-to-low`, `agentic-high-to-low`, `throughput-high-to-low`, `latency-low-to-high`, `newest`). — [Models API reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
- `min_tool_success_rate` is "a fraction in [0, 1] (e.g. 0.9 = 90% of requests finishing with a tool_calls finish reason)". The `Model` response schema has no tool-success field. I searched the schema: only the filter params exist. — [Models API reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
- `benchmarks` ("Omitted when no benchmark data is available") contains `artificial_analysis{intelligence_index, coding_index, agentic_index}` and `design_arena[]{arena, category, elo, rank, win_rate}`. — [Models API reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
- Live free-model AA indexes are sparse. Nemotron 3 Ultra free: intelligence 11.8, coding 49.3, agentic 20.1. Gemma 4 31B free: 4.4 / 43.4 / 4.2. Flint, Xing4.0 and dots-3 have no `benchmarks`. — [Live models API](https://openrouter.ai/api/v1/models)
- `expiration_date`: "The date after which the model may be removed. ISO 8601 (YYYY-MM-DD) or null". Live example: `dots-studio/dots-3-note-preview:free` expires `2026-12-31`. — [Models API reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties); [Live models API](https://openrouter.ai/api/v1/models)
- `GET /api/v1/models/user` lists "models filtered by user provider preferences, privacy settings, and guardrails" (text output by default). — [models/user reference](https://openrouter.ai/docs/api/api-reference/models/list-models-filtered-by-user-provider-preferences-privacy-settings-and-guardrails)
- Generation API: `GET /api/v1/generation?id=gen-...` returns `model`, `provider_name`, `router` (e.g. `openrouter/auto`), `latency` (ms), `generation_time` (ms), `moderation_latency`, `finish_reason`, `native_finish_reason`, `tokens_prompt`/`tokens_completion`, `native_tokens_*`, `total_cost`, `is_byok`, `streamed`, `cancelled`, `upstream_id`, and `provider_responses[]{provider_name, model_permaslug, endpoint_id, status, latency, is_byok}`, described as "including fallback attempts". — [Generation API reference](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation)
- The `X-Generation-Id` response header lets you fetch the generation record for post-mortems. — [Router Metadata docs](https://openrouter.ai/docs/guides/features/router-metadata)

### Inferences
- Rank free candidates by (a) the SDK's own observed 429 and success rates (uptime cannot supply this), (b) `uptime_last_30m`/`1d` for outages, (c) authenticated `latency_last_30m.p50` and `throughput_last_30m.p50`, and (d) `benchmarks.artificial_analysis` where present. Fetch endpoints with the API key, since anonymous calls return null perf data.
- Discovery could use `GET /api/v1/models?max_price=0&max_output_price=0&output_modalities=text&supported_parameters=tools` server-side, but keep the local checks. Better still, use `GET /api/v1/models/user` so models blocked by the account's privacy settings never enter the rotation (see Q6).
- Pick a discovery refresh rate based on `expiration_date` and churn (Q8). Daily is a reasonable floor.

### Gaps
- The meanings of the `status` enum values (0, -1, -2, -3, -5, -10) are undocumented.
- Latency units conflict. `PercentileStats` says milliseconds, the endpoint example shows `p50: 0.25` (seconds-like), and `perf_last_30m_by_workload` shows `250`. Verify against authenticated live data.
- Not tested: whether `/models/user` includes `openrouter/free` and zero-priced stealth models.

## Q6. Data and privacy for free models: logging/training, account settings that block free models, and the resulting error

### Takeaway
OpenRouter itself does not store prompts unless you opt in. Free-model hosts, however, mostly retain or may train. Only one of the 12 currently listed `:free` models (all text-output) is on the ZDR-eligible list. Account privacy toggles have separate switches for free endpoints that may train. Turning them off makes single-host free models unroutable, and the request fails with 404 `"No endpoints found matching your data policy."`. ZDR-only mode removes nearly all free models.

### Cited Findings
- OpenRouter "does not store your prompts or responses, *unless* you opt in" (private logging, or product-use for a 1% discount) or content is flagged as potential CSAM. It does store metadata such as token counts and latency. — [Data Collection docs](https://openrouter.ai/docs/guides/privacy/data-collection)
- "On your account settings page, you can set whether you would like to allow routing to providers that may train on your data… **There are separate settings for paid and free models.** … If you opt out of training… OpenRouter will not route to providers that train." — [Provider Logging docs](https://openrouter.ai/docs/guides/privacy/provider-logging)
- "If you specify provider routing in your request, but none of the providers match the level of privacy specified in your account settings, you will get an error and your request will not complete." — [FAQ](https://openrouter.ai/docs/faq)
- Three data-policy tiers: ZDR, Safety Retention (retained only for abuse prevention, not training), and Retention and Training. When OpenRouter cannot ascertain a policy, it assumes the endpoint both retains and trains. — [ZDR docs](https://openrouter.ai/docs/guides/features/zdr)
- `data_collection: "deny"` means "use only providers which do not collect user data". — [Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection)
- Live ZDR list (`/api/v1/endpoints/zdr`, 1117 endpoints, 2026-11-06): the only text `:free` model present is `xingchen-agi/xing4.0-29b-a4b:free` (host Phala, `safety_retention: false`). The other 11 `:free` models and `stealth/crouton-alpha` are absent. — [Live ZDR endpoints](https://openrouter.ai/api/v1/endpoints/zdr)
- Community (data as of 2026-10-05): of 19 free text models, 9 were on hosts marked "May train" (NVIDIA, Thinking Machines, Liquid, Stealth), 6 "Retains" (Google AI Studio 55 days, Cohere 30 days, Poolside, AtlasCloud), and 4 zero-retention (mostly Novita). Also: "Switch off 'free endpoints that may train'… It makes nine of the nineteen models unavailable", and the "free endpoints that may publish prompts" switch "gates nothing today". — [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- Stealth: `stealth/crouton-alpha` (2026-11-03): "Prompts and completions may be retained by the provider but are not used for training". Earlier stealth models in 2025 (Optimus Alpha, Cypher Alpha) said prompts were logged by the lab and may be used for training. — [Stealth models page](https://openrouter.ai/stealth); [Crouton Alpha page](https://openrouter.ai/stealth/crouton-alpha); [Cypher Alpha post (2025-07)](https://openrouter.ai/blog/new-stealth-model-cypher-alpha) (snippet via search, not fully fetched)
- Error text (community, 2025-04): `"No endpoints found matching your data policy."` (404), with the fix "Enable Model Training under Privacy Settings". Also `"No allowed providers are available for the selected model"` (404) when provider allow/ignore lists are too strict. — [Janitor AI help](https://help.janitorai.com/en/article/openrouter-error-guide-10ear52/)
- Planned and recent policy changes: from 2026-10-27, OpenRouter screens all inference, including ZDR traffic, for CSAM. From 2026-10-18, Google Vertex models require Safety Retention. — [ZDR docs](https://openrouter.ai/docs/guides/features/zdr)

### Inferences
- Treat 404 `No endpoints found matching your data policy` as a permanent per-account exclusion of that model, not a transient error. Do not retry; drop the model from rotation until settings change. Better, prevent it by discovering through `GET /api/v1/models/user`.
- Treat `No endpoints found for <model>` (404) as model removed (churn) and evict it from the cache.
- The SDK should document that free-model prompts are generally retained or possibly trained on, and expose an opt-in "zdr-only" mode. Today that mode would leave about one model plus possibly `openrouter/free`.

### Gaps
- The exact current UI labels of the free-model privacy toggles were not verified. The settings page requires login.
- The current per-provider data-policy table was not retrieved programmatically. The public endpoints schema has no `data_policy` field, though the ZDR list has `safety_retention`.

## Q7. BYOK, key-level limits (`limit`, `limit_remaining`), `is_free_tier`, and the deprecated `rate_limit`

### Takeaway
BYOK requests are not gated by the free-model daily counter, and the upstream-429 message explicitly suggests BYOK to "accumulate your rate limits". `GET /api/v1/key` exposes `free_model_daily_requests{used, limit, remaining}`. The tier is chosen by all-time credits purchased, not by `is_free_tier`. The `rate_limit` object is deprecated ("safe to ignore").

### Cited Findings
- `GET /api/v1/key` `data` fields: `label`, `limit`, `limit_reset`, `limit_remaining`, `include_byok_in_limit`, `budgets[]`, `effective_budget`, `usage`, `usage_daily/weekly/monthly`, `byok_usage*`, `is_free_tier`, `is_management_key`, `is_provisioning_key`, `free_model_daily_requests{used, limit, remaining}`, and a deprecated `rate_limit` (example `{interval: "1h", requests: 1000, note: "This field is deprecated and safe to ignore."}`). — [Get current API key](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key); [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- "Accounts and endpoints exempt from free-model limits, and BYOK requests, are not gated by it, so `remaining` reflects the tier policy rather than an enforced ceiling for them… The `limit` tier is selected by all-time credits purchased, independently of `is_free_tier`. To absorb rounding and top-up fees, the higher daily ceiling is granted starting one credit below the table's threshold [i.e. 9]; an account that has purchased fewer credits than that reports `is_free_tier: false` together with the lower daily ceiling." — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- `GET /api/v1/key` and `GET /api/v1/credits` have per-IP burst limits, and those 429s carry no `X-RateLimit-*` headers, so back off for a few seconds. — [Limits docs](https://openrouter.ai/docs/api/reference/limits)
- Per-key credit limit exhaustion and the in-flight budget surface as 402 with `error.metadata.limit_source` (`openrouter_key_limit`, `openrouter_credits`, `openrouter_in_flight_budget`). Only the in-flight case comes with `Retry-After`. — [Limits docs](https://openrouter.ai/docs/api/reference/limits); [Errors docs](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- BYOK: "When you use OpenRouter credits, your rate limits for each provider are managed by OpenRouter. Using provider keys enables direct control over rate limits". BYOK costs 5% of list price after a $25,000/month free allowance on pay-as-you-go. By default, if all BYOK keys hit rate limits, requests fall back to shared OpenRouter endpoints. — [BYOK docs](https://openrouter.ai/docs/guides/overview/auth/byok); [Pricing](https://openrouter.ai/pricing)
- A BYOK key does not exempt a provider from `data_collection` restrictions, and does not bypass ZDR unless the key declares ZDR. — [BYOK docs](https://openrouter.ai/docs/guides/overview/auth/byok)
- Upstream free 429 text: "…or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations". — [LibreChat #11480](https://github.com/danny-avila/LibreChat/discussions/11480)
- Community (2026-10): third-party guides still quote "20 requests per 10 seconds" or "200 per 10 seconds" derived from the dead `rate_limit` field. "Those numbers do not describe anything the platform enforces today." — [TrueFoundry, rate limits](https://www.truefoundry.com/blog/openrouter-rate-limits-your-free-tier-depends-on-what-youve-already-paid)

### Inferences
- Size the daily budget from `free_model_daily_requests.limit/remaining`, not from `is_free_tier` and not from `rate_limit`. Call `/key` sparingly (startup plus after a daily-cap 429), because it has its own per-IP limit.
- An optional BYOK path (for example, the user's own Google AI Studio key for Gemma) is the documented way around free-model caps. The SDK can detect it via `openrouter_metadata.is_byok` or `generation.is_byok`.

### Gaps
- Not confirmed: whether a BYOK request to a `:free` model ID is billed or counted. The docs only say BYOK is "not gated" by the free counter.

## Q8. Recent (2025–2026) free-tier changes, free-model churn, stealth models, deprecations

### Takeaway
Limits went from 20 RPM/200 RPD (documented Jan–Mar 2025) to 50 RPD, or 1000 RPD with at least 10 credits (reported April 2025), and are unchanged as of Nov 2026. Free-model churn is very high. In about one month (2026-10-05 to 2026-11-06), 9 of 19 zero-priced text models disappeared and 4 appeared. Several free variants were withdrawn while the paid base model remained. Stealth models are free, zero-priced, without a `:free` suffix, and short-lived.

### Cited Findings
- Early 2025 docs quoted "20 requests per minute and 200 requests per day" for `:free`. — [smolagents #361 (2025-01-26)](https://github.com/huggingface/smolagents/issues/361); [LiteLLM #9035 (2025-03-06)](https://github.com/BerriAI/litellm/issues/9035)
- Reported April 2025 change: free limits cut "200 -> 50 requests per day". This rests on a single community thread, seen only as a search snippet, with no official announcement found. — [r/SillyTavernAI "New OpenRouter limits" (2025-04-07)](https://www.reddit.com/r/SillyTavernAI/comments/1jtyibt/new_openrouter_limits)
- Current official limits: 20 RPM, 50 RPD, or 1000 RPD at at least 10 credits. — [Limits docs](https://openrouter.ai/docs/api/reference/limits); [Pricing](https://openrouter.ai/pricing)
- Pricing page marketing: Free plan "25+ free models" and "4 free providers". — [Pricing](https://openrouter.ai/pricing)
- My own diff of the live API (2026-11-06) against TrueFoundry's 2026-10-05 list:
  - Now: 12 `:free` models, all text-output, one of them the `nvidia/nemotron-3.5-content-safety:free` safety classifier, plus `stealth/crouton-alpha` (zero-priced) and `openrouter/free`. That makes 13 zero-priced text models versus 19 on 2026-10-05.
  - Removed: `nvidia/nemotron-3-super-120b-a12b:free`, `stealth/space-bunny-alpha`, `thinkingmachines/inkling:free`, `thinkingmachines/inkling-small:free`, `poolside/laguna-s-2.1:free`, `poolside/laguna-xs-2.1:free`, `apodex/apodex-1.1-mini:free`, `inclusionai/ling-3.0-flash-sante:free`, `qwen/qwen3.8-27b:free`. For Nemotron 3 Super, both Inklings and Qwen3.8, the paid base model is still listed.
  - Added: `recursive/flint-v1.0:free` (2026-10-22), `xingchen-agi/xing4.0-29b-a4b:free` (2026-10-17), `nvidia/nemotron-3.5-super-vl:free` (2026-10-16), `stealth/crouton-alpha` (2026-11-03).
  - Sources: [Live models API](https://openrouter.ai/api/v1/models); [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- Community (as of 2026-10-05): median free model listed for 75 days, 8 of 19 appeared in the prior 60 days, and the oldest had been listed for 207 days. `qwen/qwen3-coder:free` and `deepseek/deepseek-r1:free` (recommended in a July guide) are gone while the paid versions remain. — [TrueFoundry, free models](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you)
- Zero-priced non-`:free` models exist. On 2026-10-05, `inclusionai/ling-3.1-flash` and `stealth/space-bunny-alpha` were free without the suffix. Today it is `stealth/crouton-alpha`. — [TrueFoundry](https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you); [Live models API](https://openrouter.ai/api/v1/models)
- Stealth lifecycle (official): Ox Alpha (2026-08-20), Union Alpha (2026-09-16, revealed as Unbiased Pareto, "free stealth period has ended"), Space Bunny Alpha (2026-09-23, revealed as MiniMax M3.1 Flash, ended), Crouton Alpha (2026-11-03, currently "available at no cost"). Crouton Alpha has 1,048,576 context, accepts tools and `response_format`, and processed 1.61T tokens in 7 days. — [Stealth page](https://openrouter.ai/stealth); [Crouton Alpha](https://openrouter.ai/stealth/crouton-alpha)
- Catalog semantics: `:free` is a "catalog variant" with its own pricing, context and endpoints. "Sending a catalog suffix on a model that has no such entry does not fall back to the base model". The single-model lookup returns 404 and inference fails. — [Model Variants overview](https://openrouter.ai/docs/guides/routing/model-variants/overview)
- Live `expiration_date` is set only on `dots-studio/dots-3-note-preview:free` (2026-12-31). The other removals above had no visible advance `expiration_date` in my snapshot. — [Live models API](https://openrouter.ai/api/v1/models)

### Inferences
- The SDK's `id.endswith(":free")` filter misses free stealth models such as `stealth/crouton-alpha`, which is large, tool-capable and widely used. Consider also admitting zero-priced, text-output models from the `stealth/` author, behind an opt-in flag because of the retention policy.
- Removal usually comes without an `expiration_date` warning. Refresh discovery at least daily, and evict on 404 `No endpoints found`.
- Exclude non-chat free models such as `nvidia/nemotron-3.5-content-safety:free`, a safety classifier, using `supported_parameters` (no `tools`) or a denylist.

### Gaps
- Not confirmed: whether zero-priced non-`:free` stealth models count against the 50/1000 RPD and 20 RPM free limits. The docs tie the limits to IDs ending in `:free`.
- No official changelog entry was found for the April 2025 limit change, and no 2026 limit change was found.

## Q9. Official OpenRouter Python SDK: does it have retry or fallback features?

### Takeaway
Yes, an official SDK exists: `pip install openrouter`, v1.3.133 released 2026-11-06, Speakeasy-generated, sync and async. By default `chat.send` retries only **5XX** with exponential backoff (500 ms initial, 60 s max, exponent 1.5, up to 1 hour elapsed) and honors `Retry-After`/`retry-after-ms`. It does **not** retry 429 by default and has no client-side model rotation. Server-side fallback is only what the `models` param provides.

### Cited Findings
- Package and repo: PyPI `openrouter` 1.3.133 ("Official Python Client SDK for OpenRouter"), repo `github.com/OpenRouterTeam/python-sdk`, "stable as of v1.0", sync and async clients. — [PyPI](https://pypi.org/project/openrouter/); [Python SDK overview](https://openrouter.ai/docs/client-sdks/python/overview); [README](https://github.com/OpenRouterTeam/python-sdk)
- The default retry in `chat.py` is `utils.RetryConfig("backoff", utils.BackoffStrategy(500, 60000, 1.5, 3600000), True)` with `retry_config = (retries, ["5XX"])`. 429 is matched as an error response, not a retry. — [python-sdk `src/openrouter/chat.py`](https://github.com/OpenRouterTeam/python-sdk/blob/main/src/openrouter/chat.py)
- `RetryConfig(strategy, backoff, retry_connection_errors, status_codes_override=None)`. `status_codes_override` holds "Retryable HTTP status codes that take precedence over the per-operation defaults". `BackoffStrategy(..., jitter_ms=None)`. "When a response carries a `Retry-After` or `retry-after-ms` header, that delay is used as-is". — [python-sdk `src/openrouter/utils/retries.py`](https://github.com/OpenRouterTeam/python-sdk/blob/main/src/openrouter/utils/retries.py); [RetryConfig docs](https://openrouter.ai/docs/client-sdks/python/models/utils/retryconfig)
- The README's retries section is a placeholder (`<!-- No Retries [retries] -->`). — [README](https://github.com/OpenRouterTeam/python-sdk)

### Inferences
- The official SDK does not cover the `freeai` niche: free-model discovery, account-aware 429 classification, budget tracking and rotation. Do not depend on it. Its retry behavior is a useful reference: honor `Retry-After`/`retry-after-ms`, add jitter, and cap elapsed time. Note that its 1-hour default `max_elapsed_time` would be harmful on a 50 RPD free budget if 429 were added to `status_codes_override`.
- `freeai` (httpx) can stay independent and send the same wire parameters (`models`, `provider`, `X-OpenRouter-Metadata`).

### Gaps
- Not checked: whether async methods or `responses.send` share the same defaults. Only `chat.py` was inspected.
