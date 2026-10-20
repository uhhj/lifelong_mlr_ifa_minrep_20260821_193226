#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021/env.sh
test ! -e "$CORESET_OUT/supervisor.exit"
test ! -e "$CORESET_OUT/pipeline.exit"
exec 9>"$CORESET_OUT/supervisor.lock"
flock -n 9
echo "$$" > "$CORESET_OUT/supervisor.pid"
trap 'rc=$?; echo "$rc" > "$CORESET_OUT/supervisor.exit"' EXIT
python -u -m rctr_exp.pipeline
python -u -m rctr_exp.final_process
