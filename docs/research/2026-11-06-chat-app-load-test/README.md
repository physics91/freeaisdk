# 2026-11-06 sample chat-app load test

Evidence for quota / cooldown behavior used in README and Groq preset comments.

| File | Role |
|---|---|
| [`report.md`](report.md) | Measurement write-up |
| [`data/`](data/) | Per-run JSONL artifacts |

New bench runs from `examples/chat_app/bench.py` append to
`data/results.jsonl` by default.
