#!/usr/bin/env bash
set -uo pipefail
Q=/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017
test ! -e "$Q/wrapper.started" || exit 71
date -Is > "$Q/wrapper.started"
echo $$ > "$Q/wrapper.pid"
bash "$Q/run_queue.sh"
rc=$?
if test "$rc" = 0; then
 /data/mlr_ifa/env/bin/python -u "$Q/final_queue_audit.py" > "$Q/final_queue_audit.log" 2>&1
 rc=$?
fi
printf '%s\n' "$rc" > "$Q/wrapper.exit"
exit "$rc"
