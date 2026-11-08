#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
rm -rf dist build
echo "cleared local dist/ build/"
