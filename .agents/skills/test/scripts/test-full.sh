#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
mkdir -p artifacts/tests/full
uv sync --locked
uv run --with ruff ruff check src tests | tee artifacts/tests/full/ruff-check.log
uv run --with ruff ruff format --check src tests | tee artifacts/tests/full/ruff-format.log
uv run --with mypy mypy | tee artifacts/tests/full/mypy.log
uv run --with opentelemetry-sdk pytest -q --ignore=tests/test_live.py --junitxml=artifacts/tests/full/junit.xml | tee artifacts/tests/full/pytest.log
if [[ "${FREEAI_LIVE:-}" == "1" ]]; then
  uv run --with opentelemetry-sdk pytest -q tests/test_live.py --junitxml=artifacts/tests/full/junit-live.xml | tee artifacts/tests/full/pytest-live.log
fi
