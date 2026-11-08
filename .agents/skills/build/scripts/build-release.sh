#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
BUILD_VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-version) BUILD_VERSION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$BUILD_VERSION"
echo "$BUILD_VERSION" | grep -qE '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
rm -rf dist build
uv sync --locked
uv build
shopt -s nullglob
wheels=(dist/*"${BUILD_VERSION}"*.whl)
sdists=(dist/*"${BUILD_VERSION}"*.tar.gz)
test ${#wheels[@]} -ge 1
test ${#sdists[@]} -ge 1
