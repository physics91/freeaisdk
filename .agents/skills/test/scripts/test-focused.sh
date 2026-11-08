#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
TARGETS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$TARGETS"
mkdir -p artifacts/tests/focused
# shellcheck disable=SC2086
uv run --with opentelemetry-sdk pytest -q --junitxml=artifacts/tests/focused/junit.xml $TARGETS | tee artifacts/tests/focused/pytest.log
