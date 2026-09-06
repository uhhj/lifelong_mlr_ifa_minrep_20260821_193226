#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="$PWD/runs/paired_rollout_diagnosis"
PYTHON="${PYTHON:-/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python}"
exec 9>"$RUN/queue.lock"
flock 9
grep -q '^ROLLOUTS_COMPLETE ' "$RUN/status.txt"
"$PYTHON" scripts/summarize_paired_rollout.py
"$PYTHON" scripts/export_paired_rollout_examples.py
printf 'SUMMARY_COMPLETE %s\n' "$(date -Is)" > "$RUN/summary_status.txt"
