#!/usr/bin/env bash
set -uo pipefail
source /data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed42_20261013/source/coreset_exp/env.sh
bash "$CORESET_SOURCE/coreset_exp/pipeline.sh"
rc=$?
if test "$rc" = 0; then
  python -u -m coreset_exp.acceptance > "$CORESET_OUT/independent_acceptance.log" 2>&1
  rc=$?
fi
printf '%s\n' "$rc" > "$CORESET_OUT/supervisor.exit"
exit "$rc"
