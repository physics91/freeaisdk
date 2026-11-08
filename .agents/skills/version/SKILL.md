---
name: version
description: >
  Update freeai SemVer in VERSION as source of truth, promote CHANGELOG.md, and
  mirror pyproject.toml [project].version. Do not commit, tag, push, publish, or
  deploy. Use release for orchestration, release-gate for validation, build for
  wheels, test for quality gates, and deploy for PyPI.
---

# Version

## Ownerless Bugi Contract (Consent Required)

- `VERSION` is the release version source.
- `CHANGELOG.md` keeps `## [Unreleased]` and promoted `## [X.Y.Z]` sections.
- `pyproject.toml` `[project].version` mirrors `VERSION`.
- Release versions are bare `X.Y.Z`.
- This skill does not commit, tag, push, publish, deploy, or create GitHub Releases.
- After edits, run `release-gate` `preflight` when available.

## Default Version Contract

Same ownerless bugi rules as above for this repository. No semantic-release,
changesets, or release-please owner was detected at bootstrap.

## Procedure

1. Confirm the target SemVer (`major`/`minor`/`patch` or explicit `X.Y.Z`).
2. Work only in the dedicated task worktree required by repository Git rules.
3. Apply the bump:

```bash
export REPO_ROOT="$(pwd)"
.agents/skills/version/scripts/bump-version.sh --new-version 0.2.0
.agents/skills/version/scripts/verify-version.sh
```

4. Run release-gate preflight (see `release-gate`) before any release commit.

## Success Criteria

- `VERSION`, `CHANGELOG.md`, and `pyproject.toml` agree on the new SemVer
- `## [Unreleased]` remains present
- No git tag, commit, push, or publish was performed by this skill

## Caveats

- Repo is seeded with `VERSION`/`CHANGELOG.md` at `0.1.0`; bumps rewrite those files
- Does not rewrite historical changelog sections beyond inserting the new heading

## Boundaries

- Do not use for release tags/GitHub Releases — use `release`
- Do not use for PyPI upload — use `deploy`
- Do not use for tests/builds — use `test` / `build`
- Do not use for gate-only checks — use `release-gate`
