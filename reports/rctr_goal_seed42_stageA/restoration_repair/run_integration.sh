#!/usr/bin/env bash
set -u
R=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020/restoration_repair
test ! -e "$R/integration.started" || exit 90
date -Is > "$R/integration.started"
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
python "$R/integration_probe.py" > "$R/integration.log" 2>&1
rc=$?
echo "$rc" > "$R/integration.exit"
exit "$rc"
