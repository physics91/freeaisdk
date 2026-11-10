# Changelog

## [Unreleased]

### Fixed

- Groq: sustained traffic no longer alternates accepted requests with RPM 429s. Groq gives a request
  back per whole 2 s since the last accepted one and drops the remainder, while the SDK's rolling
  window refilled continuously and caught up after a late request. The preset now models this, and
  reads the window back from RPM 429 bodies, as no header reports it.
- Groq: qwen's 1000 output tokens per minute are enforced locally and read back from OTPM 429 bodies.

### Added

- `Limit.whole` and `RateSnapshot.whole` for windows that give quota back by whole intervals.
- `otpm()`, the `"output_tokens"` unit, `Cost.output_tokens` and `Provider.output_reserve_tokens`.

## [0.1.0]

- Alpha package aligned with pyproject.toml version 0.1.0.
