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
rm -rf dist build
uv build
uv run --with twine twine check --strict dist/*
