#!/usr/bin/env bash
set -u
R=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020/restoration_repair
test ! -e "$R/started" || exit 90
date -Is > "$R/started"
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
python "$R/repair_probe.py" > "$R/run.log" 2>&1
rc=$?
echo "$rc" > "$R/run.exit"
exit "$rc"
