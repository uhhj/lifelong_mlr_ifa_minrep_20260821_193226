#!/usr/bin/env bash
W=/data/mlr_ifa/experiment_workspaces/ZR_coreset_OBJECT_seed42_20261017
E="$W/engineering_recovery"
bash "$E/resume.sh"
rc=$?
if [ "$rc" -eq 0 ]; then
  source "$W/env.sh"
  python "$E/process_audit.py" > "$E/process_audit.log" 2>&1
  rc=$?
fi
printf '%s\n' "$rc" > "$E/supervisor.exit"
