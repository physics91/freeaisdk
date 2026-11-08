#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
uv sync --locked
