#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/experiment_workspaces/E1_protective_object_20261008/source/protective_e1/env.sh
R="$E1_OUT/engineering_recovery/20261008_import_binding"
export E1_RECOVERY_BOOTSTRAP="$R/bootstrap"
python() { PYTHONPATH="$E1_RECOVERY_BOOTSTRAP:$PYTHONPATH" /data/mlr_ifa/env/bin/python "$@"; }
export -f python
exec 9>"$E1_OUT/process.lock"
flock -n 9
printf '%s\n' "$$" > "$R/resume.pid"
trap 'code=$?; printf "%s\n" "$code" > "$R/resume.exit"' EXIT
run_step() {
 local label="$1"; shift
 printf '%s START %s\n' "$(date -Is)" "$label" >> "$R/commands.log"
 "$@" > "$R/${label}.log" 2>&1
 printf '%s EXIT 0 %s\n' "$(date -Is)" "$label" >> "$R/commands.log"
}
run_step evaluate_primary bash protective_e1/evaluate.sh primary
run_step evaluate_secondary bash protective_e1/evaluate.sh secondary
run_step finalize_readonly python -B "$R/finalize_readonly.py"
