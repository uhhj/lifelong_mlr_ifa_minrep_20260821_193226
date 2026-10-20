#!/usr/bin/env bash
set -u
W=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020
test ! -e "$W/recovery.started" || exit 90
date -Is > "$W/recovery.started"
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
python "$W/recover_asset_paths.py" > "$W/recovery.log" 2>&1
result=$?
echo "$result" > "$W/recovery.exit"
exit "$result"
