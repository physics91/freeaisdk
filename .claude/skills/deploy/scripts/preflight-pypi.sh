#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
TAG=""
PREVIOUS_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --previous-tag) PREVIOUS_TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$TAG"
test -n "$PREVIOUS_TAG"
[[ "$TAG" == v* ]] || { echo "TAG must look like vX.Y.Z" >&2; exit 1; }
git rev-parse --verify "$TAG" >/dev/null
git rev-parse --verify "$PREVIOUS_TAG" >/dev/null
test -f .github/workflows/publish.yml
command -v gh >/dev/null
uv sync --locked
uv run --with opentelemetry-sdk pytest -q --ignore=tests/test_live.py
uv build
uv run --with twine twine check --strict dist/*
