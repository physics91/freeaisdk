#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
test -d .venv
uv run freeai models --help >/dev/null
