#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/preflight.sh"
# CI mirror: include strict mypy.
uv run --with mypy mypy
uv run --with opentelemetry-sdk pytest -q --ignore=tests/test_live.py
rm -rf dist build
uv build
uv run --with twine twine check --strict dist/*
VERSION="$(tr -d '[:space:]' < VERSION)"
shopt -s nullglob
wheels=(dist/*"${VERSION}"*.whl)
sdists=(dist/*"${VERSION}"*.tar.gz)
test ${#wheels[@]} -ge 1
test ${#sdists[@]} -ge 1
