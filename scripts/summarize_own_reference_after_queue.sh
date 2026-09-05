#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER=runs/own_reference_control_driver
# The existing queue owns this lock until all jobs exit. Do not alter or restart it.
exec 9>"$DRIVER/queue.lock"
flock 9
grep -q '^COMPLETE ' "$DRIVER/status.txt"
"${PYTHON:?PYTHON required}" scripts/summarize_own_reference_control.py
date -Is > "$DRIVER/SUMMARY_COMPLETE"
