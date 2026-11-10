---
name: release
description: >
  Orchestrate freeai ownerless bugi releases: VERSION/CHANGELOG consistency,
  release/vX.Y.Z flow via repository policy and commit-work, annotated tag
  vX.Y.Z, and separately confirmed GitHub Release / PyPI publish through deploy.
  Do not use for version-only file edits (use version), gate-only checks (use
  release-gate), ordinary builds/tests (use build/test), or silent PyPI uploads.
---

# Release

## Ownerless Bugi Contract (Consent Required)

- Base branch defaults to `main`.
- Release branch shape is `release/vX.Y.Z`.
- Version source is `VERSION`; `pyproject.toml` mirrors it.
- `CHANGELOG.md` keeps `[Unreleased]` and promoted `## [X.Y.Z]`.
- Run `release-gate` `local-release` before release commits or tags.
- Release commit contains only release files and follows project commit language,
  for example `chore(release): X.Y.Z 릴리즈`.
- Annotated tag is `vX.Y.Z`.
- Push, package publish, deployment, and GitHub Release creation require explicit
  user confirmation. Prefer `deploy` profile `pypi` for PyPI via Trusted Publishing.

## Procedure

### Phase 1: Pre-flight

```bash
git status --short --branch
git branch --show-current
.agents/skills/release-gate/scripts/preflight.sh
```

Confirm the target version is not already on PyPI before publishing later.

### Phase 2: Version + gate

1. Use `version` to set `VERSION`, changelog, and `pyproject.toml`.
2. Run `release-gate` `local-release`.
3. Delegate release branch handling, intended-file staging, and the release
   commit to repository Git worktree policy and `commit-work`. This skill must
   not prescribe direct branch-creation, staging, or commit commands.

### Phase 3: Tag / publish (confirmation required)

After an authorized release commit exists on the intended branch tip:

1. Create annotated tag `vX.Y.Z` only with current explicit user confirmation.
2. Push branch/tag only with current explicit user confirmation.
3. For PyPI, use `deploy` profile `pypi` (GitHub Release → `publish.yml`) only
   with production authorization gates.

Inspect status without mutating:

```bash
git status --short
git rev-parse HEAD
git tag --list "v*"
```

## Success Criteria

- `release-gate` `local-release` passed before release commit/tag work
- Release commit (via `commit-work`) contains only release files
- Tag/publish/deploy steps occurred only after fresh confirmations
- Repository policy and commit-work owned branch/commit mutations

## Caveats

- CI/publish workflow actions are pinned by full commit SHA; treat major action upgrades as intentional PRs
- Trusted Publisher must already be configured on PyPI for this repository

## Boundaries

- Do not use for VERSION-only edits — use `version`
- Do not use for local validation only — use `release-gate`
- Do not use for dry-run dist checks without release intent — use `deploy` `pypi-dry-run` or `build`
- Do not use for pytest — use `test`
