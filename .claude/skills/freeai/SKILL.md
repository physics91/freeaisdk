---
name: freeai
description: >
  Use free-llm-sdk (import name freeai) for free-tier LLM chat with automatic
  failover and 429 avoidance. Use when installing or calling freeai, free-llm-sdk,
  FreeAI.from_env, freeai serve, or replacing OpenAI base_url with a free stack.
---

# freeai (free-llm-sdk)

## Hard facts

- **PyPI name:** `free-llm-sdk` — not `freeai`, not `freeaisdk`
- **Import / CLI:** `freeai`
- **Python:** `>=3.12`
- **Runtime dep:** `httpx` only
- **Default:** free tiers + anonymous OVHcloud (no key required). Do not open trial/paid tiers unless asked.

## Install

```bash
uv add free-llm-sdk
# or
pip install free-llm-sdk
```

Optional OpenTelemetry: `uv add "free-llm-sdk[otel]"`

## Prefer these paths (in order)

### 1) One-liner

```python
from freeai import chat

print(chat("Answer in one sentence: what is GIL?"))
```

### 2) Env-based client

```python
from freeai import FreeAI

with FreeAI.from_env() as ai:  # or AsyncFreeAI.from_env()
    r = ai.chat("hi", system="Be brief", max_wait=30, deadline=60)
    print(r.text, r.provider, r.model)
```

Keys picked up when present (free tier only by default):

| Preset | Env |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Cloudflare | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| Z.ai | `ZAI_API_KEY` |
| SambaNova | `SAMBANOVA_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| Kilo | `KILO_API_KEY` (or anonymous with `include_keyless=True`) |
| Requesty | `REQUESTY_API_KEY` |
| OVHcloud | none (anonymous, on by default) |

- Own keys only: `FreeAI.from_env(include_keyless=False)` or `freeai --no-keyless …`
- Faster answers: `prefer=["groq:*"]` or a short `deadline=…`
- Do **not** pass `tiers=("free","trial","paid")` unless the user accepts billing risk (`BillingWarning`).

### 3) Drop-in OpenAI base URL

```bash
freeai serve   # http://127.0.0.1:8787/v1
```

Point any OpenAI-compatible client at that base URL. Set client `max_retries=0` — failover is already inside the proxy.

```python
from freeai.compat import OpenAI

client = OpenAI()  # wraps FreeAI.from_env()
```

## Do not

- `pip install freeai` / `uv add freeai` (wrong package on PyPI)
- Assume import name equals PyPI name
- Enable trial/paid tiers by default
- Log API keys or full prompts in agent traces

## More detail

- Short guide: `docs/agent/guide.md`
- Index: `/llms.txt`
- Full human docs: `README.md`
