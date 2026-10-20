#!/usr/bin/env bash
set -u
W=/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA2_20261020
OLD=/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020
O="$OLD/engineering_recovery/dedicated_env_v1"
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
cd "$W"
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$W:$PYTHONPATH"
test "$(cat "$OLD/runner.exit")" = 1 || exit 90
test ! -e "$O" || exit 91
mkdir -p "$O"
printf '%s\n' "$$" > "$O/supervisor.pid"
date -Iseconds > "$O/started"
python -u "$W/recovery_runner.py" > "$O/runner.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$O/runner.exit"
if [ "$rc" -eq 0 ]; then
  python -u "$W/analyze.py" "$O" > "$O/analyze.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$O/analyze.exit"
fi
printf '%s\n' "$rc" > "$O/supervisor.exit"
exit "$rc"
