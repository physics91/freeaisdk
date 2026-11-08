---
name: build
description: >
  Build the freeai Python package with uv (sync --locked, uv build) and verify
  wheel/sdist artifacts under dist/. Use for local package builds and release
  artifact preparation. Do not use for running tests (use test), publishing to
  PyPI (use deploy), version bumps (use version), or release orchestration
  (use release / release-gate).
---

# Build

## Profile Routing Contract

Keep project-specific build routes in skill-root `profiles.json`. Validate the
document against the repository root before selecting a route.

1. If the user supplied a profile ID, require an exact key match in `profiles`.
2. If omitted, use only the validated `default_profile` (`release`).
3. Read the selected profile object directly from the `profiles` map. Do not
   branch on known profile names.
4. Resolve only declared `required_inputs`, preserve argument order, and never
   store secrets in `profiles.json`.
5. Run the selected tracked `entrypoint`, then its `verification`, and confirm
   every declared artifact path.

## Detected Stack

- Ecosystem: Python 3.12+ library (`pyproject.toml`, `uv_build`, `uv.lock`)
- Install/sync: `uv sync --locked`
- Build command: `uv build`
- Artifacts: `dist/*.whl`, `dist/*.tar.gz`
- Cache key: `uv.lock` (uv store under `~/.cache/uv`)

Sources (retrieved 2026-11-07): https://docs.astral.sh/uv/concepts/projects/sync/ ;
project files `pyproject.toml`, `.github/workflows/publish.yml`.

## Procedure

### development

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/build/scripts/build-development.sh
.agents/skills/build/scripts/verify-development-build.sh
```

### release (default)

Resolve `BUILD_VERSION` from `VERSION` (required once the ownerless contract is
seeded). The release build script also asserts wheel/sdist names contain that
SemVer.

```bash
export REPO_ROOT="$(pwd)"
BUILD_VERSION="$(tr -d '[:space:]' < VERSION)"
.agents/skills/build/scripts/build-release.sh --build-version "$BUILD_VERSION"
.agents/skills/build/scripts/verify-release-build.sh
```

## Success Criteria

- Selected script exits with exit code 0
- `development`: `.venv` exists and `uv run freeai models --help` succeeds
- `release`: at least one non-empty wheel and sdist exist under `dist/`

## Caveats

- Requires `uv` on PATH matching the project's build-system pin (`uv_build>=0.9.28,<0.10.0`)
- Does not run the test suite, ruff, or mypy
- Does not upload packages
- `VERSION` is seeded at `0.1.0` with `CHANGELOG.md`; keep them mirrored via `version`

## Boundaries

- Do not use for pytest/ruff/mypy — use `test`
- Do not use for PyPI publication — use `deploy`
- Do not use for SemVer/changelog edits — use `version`
- Do not use for tags/GitHub Releases — use `release`
