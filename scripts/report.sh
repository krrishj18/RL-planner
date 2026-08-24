#!/usr/bin/env bash
# Print summary/eval files from fetched runs (use with --fetch <tag>).
set -uo pipefail
find runs -name 'summary.md' -o -name 'summary.csv' | sort | while read -r f; do
  echo "===== $f ====="; cat "$f"; echo
done
