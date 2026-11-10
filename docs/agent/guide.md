# freeai for AI agents

Short contract for agents integrating **free-llm-sdk** (import/CLI: **freeai**).

## Install and import

```bash
uv add free-llm-sdk
```

```python
from freeai import chat, FreeAI, AsyncFreeAI
```

PyPI name ≠ import name. Never install `freeai` or `freeaisdk` from PyPI for this project.

## Minimal usage

```python
from freeai import chat
print(chat("Say hello in one short sentence."))
```

No API key is required: anonymous OVHcloud is included by default. With keys in the environment, `chat()` / `FreeAI.from_env()` also use those free-tier presets.

## Client options agents usually need

```python
from freeai import FreeAI

ai = FreeAI.from_env(
    prefer=["groq:*"],       # optional: bias to a fast provider
    include_keyless=False,   # optional: own keys only (no anonymous OVHcloud)
    max_wait=30,             # wait for quota / cooldown
    deadline=60,             # whole-call time budget
)
print(ai.chat("hi", system="Be brief").text)
ai.close()
```

Async: `async with AsyncFreeAI.from_env() as ai: ...`

## OpenAI-compatible path

```bash
freeai serve
# base URL: http://127.0.0.1:8787/v1
```

Or in-process:

```python
from freeai.compat import OpenAI
client = OpenAI()
```

When using the proxy from another OpenAI SDK, set `max_retries=0`.

## Environment variables (free presets)

| Preset | Env |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Cloudflare | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| Z.ai | `ZAI_API_KEY` |
| SambaNova | `SAMBANOVA_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| Kilo | `KILO_API_KEY` |
| Requesty | `REQUESTY_API_KEY` |
| OVHcloud | _(none)_ |

Default `from_env()` only enables **free** tiers. Trial/paid need an explicit `tiers=...` and raise `BillingWarning`.

## Common mistakes

1. Installing `freeai` from PyPI (unrelated package).
2. Importing `free_llm_sdk` (wrong; import is `freeai`).
3. Opening trial/paid tiers without user approval.
4. Retrying aggressively on top of `freeai serve` (double retry / failover).

## Copy this skill into another repo

```bash
mkdir -p .agents/skills/freeai
curl -fsSL \
  https://raw.githubusercontent.com/physics91/freeaisdk/main/.agents/skills/freeai/SKILL.md \
  -o .agents/skills/freeai/SKILL.md
```

Cursor users can place the same file under `.cursor/skills/freeai/SKILL.md` if that is the project skill root.
