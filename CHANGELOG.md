# Changelog

## [Unreleased]

### Changed

- The default `strategy` is now `"quota"` (was `"priority"`). Among the candidates that can be sent
  now, the one whose quota comes back soonest is used, so allowances that refill every minute are
  spent before daily ones and the daily ones are still there when demand peaks. The order providers
  and models are listed in no longer decides by itself: it breaks ties, and `prefer`, an explicit
  `model` and routers-last still come first. What to expect: more requests on platforms that only
  limit per minute, which tend to be slower (anonymous OVHcloud) or train on prompts (NVIDIA, Kilo;
  `allow_training=False` excludes those); a provider that declares no tier is used after the free
  ones; and 1 to 2 ms more per call with some fifty candidates. `strategy="priority"` restores the
  old order.
- `from_env()` (and with it `chat()`, `stream()`, the CLI and the proxy) now uses OVHcloud's anonymous
  endpoints without being asked: nine models at 2 requests a minute each, on a platform that states
  it never trains on prompts. It therefore also works with no key at all, and prompts go to a
  platform you have no account with. `include_keyless=False` (`--no-keyless`) uses only your own
  keys, as before; `include_keyless=True` (`--keyless`) still adds Kilo, whose free models may train
  on prompts. With the default strategy these endpoints are used first, so answers are slower than
  from Groq: 0.6 to 3.5 s against 0.3 to 0.6 s in eight requests on 2026-11-10, and the load test of
  2026-11-06 (94 attempts) had OVHcloud's first token at 7 to 16 s and whole answers at 24 to 29 s
  at the 95th percentile. `prefer=["groq:*"]`, a `deadline` or `include_keyless=False` keep it fast.
- `adapt_limits` is on by default (experimental, checked against mock platforms only): a request is
  sent beyond a daily or monthly request limit that is declared as an estimate (Gemini's daily
  requests) to find its real size. `adapt_limits=False` keeps the declared numbers.

### Fixed

- Groq: sustained traffic no longer alternates accepted requests with RPM 429s. Groq gives a request
  back per whole 2 s since the last accepted one and drops the remainder, while the SDK's rolling
  window refilled continuously and caught up after a late request. The preset now models this, and
  reads the window back from RPM 429 bodies, as no header reports it.
- Groq: qwen's 1000 output tokens per minute are enforced locally and read back from OTPM 429 bodies.

- A stored quota row kept the size its limit declared when the row was written, so changing a
  declared limit had no effect on existing state (rows of rolling windows never expire). Rows now
  keep a size only when the platform reported it; sizes copied by older versions are ignored.
- A request that is still being answered was counted twice whenever another response's headers
  already included it. Reservations made `freeai.limits.ACCEPT_MARGIN` (6 s) or more before the
  request a reading came back with are no longer added to that reading.
- One window declared twice (a preset's default and the caller's `rpm=`, e.g. `NVIDIA(rpm=20)`)
  charged every request twice in the same row; `rpm=10` on a preset declaring 40 allowed 20. The
  later declaration now replaces the earlier one.
- A commit that failed while reserving a model left its concurrency slot taken for good.
- Token headers of a request sent earlier but answered later no longer replace a later reading.
  Groq computes the headers of a non-streamed answer when it accepts the request and sends them
  with the body (measured 2026-11-10), so such a reading does not know what later requests used.

### Added

- `strategy="quota"`: among the candidates that can be sent now, pick the one whose quota comes
  back soonest, so allowances that refill every minute are spent before daily ones. `prefer`, an
  explicit `model` and routers-last still come first; free tiers are used before providers of
  unknown tier, then trial, then paid.
- `Provider.uncounted` names the window of an allowance the SDK cannot count (Cloudflare's daily
  neurons, Z.ai's usage limit, a keyed Kilo account), so the quota strategy does not treat such a
  provider as unlimited.
- `max_wait` per call on `chat()` and `stream()`, and `batch()` on both clients: many prompts with
  bounded concurrency, waiting for quota, results in input order with errors in place. The proxy
  lets a request lower `max_wait` but not raise it.
- `adapt_limits` (experimental) and `Limit.estimate`: request limits per day or month declared from
  unofficial sources (Gemini's daily requests) are probed when full and raised when the probe is
  accepted, and lowered when the platform reports the allowance exhausted although the window had
  room. Windows the platform reports are never adapted, nor are estimates per minute or hour.
- `FreeAI.audit()`, `AsyncFreeAI.audit()` and `freeai audit`: compare pinned model lists with each
  platform's catalog (read-only).
- CLI options `--strategy`, `--max-wait` and `--adapt-limits` / `--no-adapt-limits`.
- OVHcloud: `Qwen3.6-27B`, `Qwen2.5-VL-72B-Instruct` and `Qwen3.5-9B` (found by the audit; each
  answered an anonymous request with its own 2 per minute on 2026-11-10).
- `Limit.whole` and `RateSnapshot.whole` for windows that give quota back by whole intervals.
- `otpm()`, the `"output_tokens"` unit, `Cost.output_tokens` and `Provider.output_reserve_tokens`.

## [0.1.0]

- Alpha package aligned with pyproject.toml version 0.1.0.
