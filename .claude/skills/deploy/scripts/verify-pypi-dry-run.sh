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
VERSION="${TAG#v}"
shopt -s nullglob
wheels=(dist/*"${VERSION}"*.whl)
sdists=(dist/*"${VERSION}"*.tar.gz)
test ${#wheels[@]} -ge 1
test ${#sdists[@]} -ge 1
echo "dry-run artifacts ok for $TAG"
