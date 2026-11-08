---
name: release-gate
description: >
  Run local freeai release readiness validation (version/changelog/pyproject
  consistency, ruff, mypy, import smoke, pytest offline, uv build, twine check).
  Do not commit, tag, push, publish, or deploy. Use version for file edits,
  release for orchestration, deploy for PyPI, build/test for ordinary loops.
disable-model-invocation: true
---

# Release Gate

## Overview

Local validation only for release readiness. Companion to ownerless bugi
`version` / `release` skills. Must not mutate remote systems.

## Safety

- SAFE-001: Do not commit, tag, push, deploy, publish packages, create GitHub
  Releases, mutate remote infrastructure, or send notifications.
- SAFE-002: Implicit invocation disabled in `agents/openai.yaml`.

## Routes

- `preflight`: VERSION/CHANGELOG/pyproject match, ruff lint/format, import smoke
- `local-release`: `preflight` plus mypy, offline pytest, `uv build`, `twine check`
- `production-readiness`: not automated here; use explicit live tests and PyPI
  Trusted Publisher checklist when operators request it

## Procedure

1. Confirm the task is release readiness validation.
2. Run `preflight` for fast feedback.
3. Run `local-release` before release commits or tags.
4. Record skipped layers (for example live network tests) as residual risk.
5. Stop on the first nonzero exit.

## Commands

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/release-gate/scripts/preflight.sh
.agents/skills/release-gate/scripts/local-release.sh
```

## Success Criteria

- Version source and `pyproject.toml` mirror match
- Changelog contains `[Unreleased]` and the target version section
- Lint/type/import checks pass
- Offline pytest and package artifact checks pass for `local-release`

## Boundaries

Do not use this skill for version file edits, release commits/tags, pushes,
GitHub Releases, or PyPI uploads.


- BOUND-001: Use `version` for VERSION/changelog/metadata edits
- BOUND-002: Use `release` for release branches, commits, tags, pushes, GitHub Releases
- BOUND-003: Use `deploy` for PyPI dry-run or production publish

## Caveats

- Requires seeded `VERSION` and `CHANGELOG.md` (bootstrap seeds `0.1.0`)
- Offline pytest only; live network tests stay opt-in via `FREEAI_LIVE=1`
- `preflight` omits mypy for speed; `local-release` includes the CI mypy gate

## Troubleshooting

- Version mismatch: run `version` and re-verify
- Missing changelog section: promote `[Unreleased]` notes into `## [X.Y.Z]`
- Twine/build failure: inspect `dist/` and `pyproject.toml` build backend pin
