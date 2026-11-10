---
name: deploy
description: >
  Deploy freeai package artifacts: default pypi-dry-run builds and twine-checks
  dist/ without upload; production pypi publishes via GitHub Release so
  publish.yml Trusted Publishing uploads to PyPI. Do not use for ordinary
  builds (use build), tests (use test), SemVer edits (use version), or release
  branch/tag orchestration without deploy intent (use release / release-gate).
disable-model-invocation: true
---

# Deploy

## Profile Routing Contract

Keep every deployment route in skill-root `profiles.json` from the accepted
bootstrap intake. Validate against the repository root before selecting.

1. Exact profile ID match when supplied; otherwise use non-production
   `default_profile` (`pypi-dry-run`).
2. Read the selected object from `profiles` without name-based branching.
3. Production (`environment: production`) requires explicit production intent.
4. Resolve `required_inputs`, run `preflight`, then seek production authority.
5. Immediately before a production `entrypoint`, check repository permission and
   obtain current explicit user confirmation. Do not reuse earlier confirmation.
6. Invoke tracked `entrypoint`, then `verification`. On failure, use `rollback`
   only with the authority that operation requires.

## Deployment Platform Matrix

| Detected Config | Platform | Entrypoint | Rollback | Verification | Default |
|----------------|----------|------------|----------|--------------|---------|
| `.github/workflows/publish.yml` + PyPI Trusted Publishing | PyPI via GitHub Release | `gh release create` triggers OIDC publish | Keep prior tag; manual yank/hotfix | `pypi.org/pypi/free-llm-sdk/<ver>/json` | `pypi-dry-run` |

Sources (retrieved 2026-11-07): https://docs.pypi.org/trusted-publishers/using-a-publisher/ ;
https://github.com/pypa/gh-action-pypi-publish ; project `publish.yml`.

## Procedure

### pypi-dry-run (default, non-production)

```bash
export REPO_ROOT="$(pwd)"
TAG=v0.1.0
.agents/skills/deploy/scripts/preflight-pypi-dry-run.sh --tag "$TAG"
.agents/skills/deploy/scripts/deploy-pypi-dry-run.sh --tag "$TAG"
.agents/skills/deploy/scripts/verify-pypi-dry-run.sh --tag "$TAG"
```

### pypi (production)

Requires explicit production request, repository permission, and fresh
confirmation after preflight.

```bash
export REPO_ROOT="$(pwd)"
TAG=v0.1.0
PREVIOUS_TAG=v0.0.0
.agents/skills/deploy/scripts/preflight-pypi.sh --tag "$TAG" --previous-tag "$PREVIOUS_TAG"
.agents/skills/deploy/scripts/deploy-pypi.sh --tag "$TAG"
.agents/skills/deploy/scripts/verify-pypi.sh --tag "$TAG"
```

## Success Criteria

- Dry-run: twine check passes and versioned files exist under `dist/`
- Production: GitHub Release exists and PyPI JSON smoke for `free-llm-sdk==VERSION` succeeds
- Rollback guidance printed with previous tag retained

## Caveats

- `pypi` preflight requires existing local tags for both `TAG` and `PREVIOUS_TAG`; for the first production publish, create/push `TAG` first and set `PREVIOUS_TAG` to the nearest prior annotated tag (or an agreed baseline tag)
- `gh release create --verify-tag` needs `TAG` to already exist locally
- Production mutates GitHub Releases and triggers PyPI upload — never run by implicit routing
- Trusted Publisher one-time setup on pypi.org must already map workflow `publish.yml` / environment `pypi`
- PyPI eventual consistency may delay verification briefly

## Boundaries

- Do not use for local wheel-only builds without deploy intent — use `build`
- Do not use for VERSION/CHANGELOG — use `version`
- Do not use for release branch commits — use `release`
- Do not use for gate validation only — use `release-gate`
