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
# Publishes GitHub Release; Trusted Publishing workflow publish.yml uploads to PyPI.
gh release create "$TAG" --verify-tag --generate-notes
