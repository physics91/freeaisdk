#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
PREVIOUS_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --previous-tag) PREVIOUS_TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$PREVIOUS_TAG"
echo "PyPI yank is manual on pypi.org for the bad version."
echo "Keep consumers on ${PREVIOUS_TAG} / free-llm-sdk==${PREVIOUS_TAG#v}."
echo "If needed, cut a hotfix release from a new patch tag after fixes."
gh release view "$PREVIOUS_TAG" >/dev/null
