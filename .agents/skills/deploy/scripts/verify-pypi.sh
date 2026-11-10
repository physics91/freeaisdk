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
gh release view "$TAG" >/dev/null
# Smoke: package page / simple index eventually lists the version.
curl -fsSL "https://pypi.org/pypi/free-llm-sdk/${VERSION}/json" >/dev/null
echo "PyPI lists free-llm-sdk==${VERSION}"
