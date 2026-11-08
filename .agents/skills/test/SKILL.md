---
name: test
description: >
  Run freeai quality gates with uv/pytest/ruff/mypy using focused, affected, or
  full profiles. Default is affected (not full). Do not use for package builds
  (use build), PyPI publish (use deploy), SemVer edits (use version), or
  release tags (use release / release-gate).
---

# Test

## Production-Ready Test Contract

- Copy-pasteable commands use `uv`, `pytest`, `ruff`, and `mypy` exactly as in
  `.github/workflows/ci.yml`.
- Deterministic state: unit/integration tests use `tests/conftest.py` factories
  (`catalog_model`, `ok`, `error`) and httpx mocks; no shared mutable DB beyond
  per-test temp paths.
- Direct target setup: reuse `catalog_model` / `completion` / `ok` / `error`
  helpers instead of replaying live provider flows for offline tests.
- CI-ready: focused, affected, and full commands write JUnit/logs under
  `artifacts/tests/<profile>/`.
- Environment safety: default routes never set `FREEAI_LIVE=1`. Live network
  checks require explicit env and keys.
- Failure signal quality: pytest stdout plus `junit.xml` under artifacts.
- Flake control: no arbitrary sleeps in offline suite; live OVH path may wait
  on rate limits — quarantine live failures separately from offline CI.
- Coverage of critical paths: routing/failover, limits/cooldown, HTTP client,
  presets/catalog, async/stream, and CLI entry via package import.
- Operational blockers: see Production Readiness Blockers for live-only gaps.

## Test Scope Router

Routes live in `profiles.json`. Select an exact profile ID when named;
otherwise use `default_profile` (`affected`). Never default to `full`.

Escalate to `full` only for an independent reason: explicit request, CI/release/
deploy/scheduled gate, broad contract/lockfile/test-infra risk, or
unavailable/uncertain affected selection. A pull request or merge alone is
never a full-suite reason. A focused/affected failure is evidence to diagnose,
not an automatic full rerun.

## Detected Stack

- `pytest` via `[dependency-groups] dev` and CI `uv run --with opentelemetry-sdk pytest -q`
- Lint: `ruff check` / `ruff format --check` on `src tests`
- Types: `mypy` (strict on `freeai`)
- Live optional: `tests/test_live.py` gated by `FREEAI_LIVE=1`

Sources (retrieved 2026-11-07): https://docs.astral.sh/uv/ ;
`.github/workflows/ci.yml`, `tests/conftest.py`.

## Procedure

### Run Focused Tests

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/test/scripts/test-focused.sh --targets "tests/test_routing.py"
```

### Run Affected Tests (default)

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/test/scripts/test-affected.sh --base origin/main --head HEAD
```

### Full Test Suite

Invoke only for an allowed full context and an independent escalation reason.
A merge is never a full-suite reason.

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/test/scripts/test-full.sh
```

CI mirror (same as workflow job):

```bash
uv sync --locked
uv run --with ruff ruff check src tests
uv run --with ruff ruff format --check src tests
uv run --with mypy mypy
uv run --with opentelemetry-sdk pytest -q
```

### Live network (explicit only)

```bash
FREEAI_LIVE=1 OPENROUTER_API_KEY=... uv run pytest -q tests/test_live.py
```

## Test Environment Setup

- setup: `uv sync --locked`
- readiness: `uv run freeai models --help`
- seed/fixture: rely on `tests/conftest.py` helpers; no DB seed required
- teardown: pytest temp dirs auto-clean; remove `artifacts/tests/` when desired
- env: no secrets for offline suite; live needs `FREEAI_LIVE=1` and provider keys
- artifacts: `artifacts/tests/{focused,affected,full}/`

## Target State Setup

### Offline chat/failover behavior

- Target behavior: `FreeAI.chat` returns text or classified `ExhaustedError`
- Required state: mock catalog + httpx responses from `catalog_model` / `ok` / `error`
- Helper: `tests/conftest.py` factories
- Precondition assert: constructed `FreeAI([...])` uses injected transport
- Exercise: run `tests/test_failover.py` / `tests/test_routing.py` via focused profile
- Cleanup: none beyond process exit

### Live OpenRouter smoke

- Target behavior: real catalog/chat against OpenRouter
- Required state: network + optional `OPENROUTER_API_KEY`
- Helper: set `FREEAI_LIVE=1` only; do not fake provider responses
- Blocker if keys missing: skip chat case; catalog may still run anonymously where supported

## Success Criteria

- Selected profile exits 0
- Artifacts written under the profile's `artifacts/tests/<id>/` path
- Offline default never requires live credentials

## Caveats

- JUnit/logs under `artifacts/tests/` are gitignored via root `artifacts/`
- `test-full` matches CI, including `ruff format --check` and `mypy`
- Affected selector confidence is medium; shared core modules often escalate
- `pytest` in publish workflow omits ruff/mypy; full profile matches CI more closely
- Live tests hit real rate limits (e.g. OVH 2 req/min)

## Production Readiness Blockers

| Blocker | Operational risk | Smallest next action |
|---------|------------------|----------------------|
| Live suite needs keys/network | Offline CI cannot prove provider auth/chat | Run `FREEAI_LIVE=1` explicitly before release when providers change |
| No dependency-graph affected tool | May under/over select tests | Keep medium-confidence heuristic; escalate on shared-module edits |

## Boundaries

- Do not use for `uv build` — use `build`
- Do not use for PyPI upload — use `deploy`
- Do not use for VERSION/CHANGELOG edits — use `version`
- Do not use for tags/GitHub Releases — use `release`
