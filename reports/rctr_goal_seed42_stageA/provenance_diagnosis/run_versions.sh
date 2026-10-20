#!/bin/bash
set -u
W=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis
O=/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
test ! -e "$O/controller.started" || exit 99
date -Is > "$O/controller.started"
printf '%s\n' "$$" > "$O/controller.pid"
python -m pip install --no-deps --target "$W/mujoco235" mujoco==2.3.5 > "$O/install235.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$O/install235.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
python "$W/version_probe.py" > "$O/probe323.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$O/probe323.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
PYTHONPATH="$W/mujoco235:${PYTHONPATH:-}" python "$W/version_probe.py" > "$O/probe235.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$O/probe235.exit"
printf '%s\n' "$rc" > "$O/controller.exit"
exit "$rc"
