#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
shopt -s nullglob
wheels=(dist/*.whl)
sdists=(dist/*.tar.gz)
test ${#wheels[@]} -ge 1
test ${#sdists[@]} -ge 1
for f in "${wheels[@]}" "${sdists[@]}"; do
  test -s "$f"
done
if [[ -f VERSION ]]; then
  VERSION="$(tr -d '[:space:]' < VERSION)"
  matched_wheels=(dist/*"${VERSION}"*.whl)
  matched_sdists=(dist/*"${VERSION}"*.tar.gz)
  test ${#matched_wheels[@]} -ge 1
  test ${#matched_sdists[@]} -ge 1
fi
