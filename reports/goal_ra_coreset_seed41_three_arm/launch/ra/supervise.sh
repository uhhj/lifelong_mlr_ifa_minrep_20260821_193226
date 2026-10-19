#!/usr/bin/env bash
set -u
W=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019/ra
source "$W/env.sh"
test ! -e "$CORESET_OUT/supervisor.started" || exit 92
date -Iseconds > "$CORESET_OUT/supervisor.started"
echo $$ > "$CORESET_OUT/supervisor.pid"
/data/mlr_ifa/env/bin/python -u "$W/pipeline.py" > "$CORESET_OUT/supervisor.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  /data/mlr_ifa/env/bin/python "$W/final_process_audit.py" > "$CORESET_OUT/process_acceptance.log" 2>&1
  rc=$?
fi
echo "$rc" > "$CORESET_OUT/supervisor.exit"
exit "$rc"
