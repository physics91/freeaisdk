#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-.}"
NEW_VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --new-version) NEW_VERSION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
test -n "$NEW_VERSION"
echo "$NEW_VERSION" | grep -qE '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'

printf '%s\n' "$NEW_VERSION" > VERSION

if [[ ! -f CHANGELOG.md ]]; then
  cat > CHANGELOG.md <<EOF
# Changelog

## [Unreleased]

## [${NEW_VERSION}]

- Initial VERSION-managed release notes.
EOF
else
  if ! grep -q '^## \[Unreleased\]' CHANGELOG.md; then
    echo "CHANGELOG.md missing ## [Unreleased]" >&2
    exit 1
  fi
  if grep -q "^## \[${NEW_VERSION}\]" CHANGELOG.md; then
    echo "CHANGELOG already has ## [${NEW_VERSION}]" >&2
    exit 1
  fi
  tmp="$(mktemp)"
  awk -v ver="$NEW_VERSION" '
    /^## \[Unreleased\]$/ {
      print
      print ""
      print "## [" ver "]"
      print ""
      next
    }
    {print}
  ' CHANGELOG.md > "$tmp"
  mv "$tmp" CHANGELOG.md
fi

tmp="$(mktemp)"
awk -v ver="$NEW_VERSION" '
  BEGIN {in_project=0}
  /^\[project\]$/ {in_project=1; print; next}
  /^\[/ {in_project=0}
  in_project && /^version[[:space:]]*=/ {
    printf "version = \"%s\"\n", ver
    next
  }
  {print}
' pyproject.toml > "$tmp"
mv "$tmp" pyproject.toml
