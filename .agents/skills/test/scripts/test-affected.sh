#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
BASE_REF=""
HEAD_REF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE_REF="$2"; shift 2 ;;
    --head) HEAD_REF="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$BASE_REF"
test -n "$HEAD_REF"
mkdir -p artifacts/tests/affected
mapfile -t CHANGED < <(git diff --name-only "$BASE_REF"..."$HEAD_REF")
if [[ ${#CHANGED[@]} -eq 0 ]]; then
  echo "no changed files between refs; running smoke presets suite" | tee artifacts/tests/affected/pytest.log
  uv run --with opentelemetry-sdk pytest -q tests/test_presets.py --junitxml=artifacts/tests/affected/junit.xml | tee -a artifacts/tests/affected/pytest.log
  exit 0
fi
BROAD=0
declare -a TARGETS=()
for path in "${CHANGED[@]}"; do
  case "$path" in
    tests/test_live.py) ;;
    tests/*.py) TARGETS+=("$path") ;;
    pyproject.toml|uv.lock|.github/workflows/*|tests/conftest.py)
      BROAD=1
      ;;
    src/freeai/*.py|src/freeai/**/*.py)
      base="$(basename "$path" .py)"
      base="${base#_}"
      candidate="tests/test_${base}.py"
      if [[ -f "$candidate" ]]; then
        TARGETS+=("$candidate")
      else
        BROAD=1
      fi
      ;;
    src/*)
      BROAD=1
      ;;
  esac
done
if [[ "$BROAD" -eq 1 ]]; then
  echo "affected selector uncertain; escalate to full independently" | tee artifacts/tests/affected/selector-note.txt
  uv run --with ruff ruff check src tests | tee artifacts/tests/affected/ruff.log
  uv run --with ruff ruff format --check src tests | tee -a artifacts/tests/affected/ruff.log
  uv run --with mypy mypy | tee artifacts/tests/affected/mypy.log
  uv run --with opentelemetry-sdk pytest -q --ignore=tests/test_live.py --junitxml=artifacts/tests/affected/junit.xml | tee artifacts/tests/affected/pytest.log
  exit 0
fi
# unique targets
declare -A SEEN=()
declare -a UNIQUE=()
for t in "${TARGETS[@]}"; do
  [[ -n "${SEEN[$t]:-}" ]] && continue
  SEEN[$t]=1
  UNIQUE+=("$t")
done
if [[ ${#UNIQUE[@]} -eq 0 ]]; then
  echo "no mapped unit tests for changes; lint only" | tee artifacts/tests/affected/pytest.log
  uv run --with ruff ruff check src tests | tee artifacts/tests/affected/ruff.log
  exit 0
fi
uv run --with opentelemetry-sdk pytest -q --junitxml=artifacts/tests/affected/junit.xml "${UNIQUE[@]}" | tee artifacts/tests/affected/pytest.log
