#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
test -f VERSION
CURRENT="$(tr -d '[:space:]' < VERSION)"
echo "$CURRENT" | grep -qE '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
grep -q '^## \[Unreleased\]' CHANGELOG.md
grep -q "^## \[${CURRENT}\]" CHANGELOG.md
grep -E "^version[[:space:]]*=[[:space:]]*\"${CURRENT}\"" pyproject.toml
