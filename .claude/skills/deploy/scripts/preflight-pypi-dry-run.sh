#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$TAG"
[[ "$TAG" == v* ]] || { echo "TAG must look like vX.Y.Z" >&2; exit 1; }
test -f pyproject.toml
uv sync --locked
uv run --with opentelemetry-sdk pytest -q --ignore=tests/test_live.py
