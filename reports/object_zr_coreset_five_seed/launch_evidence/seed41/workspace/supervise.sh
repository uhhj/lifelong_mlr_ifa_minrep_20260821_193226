#!/usr/bin/env bash
W=/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017/seed41
bash "$W/pipeline.sh"
rc=$?
if [ "$rc" -eq 0 ]; then
  source "$W/env.sh"
  python "$W/final_process_audit.py" > "$ZR_RUN/final_process_audit.log" 2>&1
  rc=$?
fi
printf '%s\n' "$rc" > /root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed41_20261017/supervisor.exit
