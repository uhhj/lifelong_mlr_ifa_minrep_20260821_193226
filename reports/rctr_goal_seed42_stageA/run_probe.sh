#!/usr/bin/env bash
set -u
W=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020
test ! -e "$W/probe.started" || exit 90
date -Is > "$W/probe.started"
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
python "$W/state_probe.py" > "$W/probe.log" 2>&1
result=$?
echo "$result" > "$W/probe.exit"
exit "$result"
