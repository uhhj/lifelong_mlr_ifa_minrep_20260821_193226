#!/usr/bin/env bash
set -uo pipefail
source /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed44/source/coreset_exp/env.sh
bash "$CORESET_SOURCE/coreset_exp/pipeline.sh"
rc=$?
if test "$rc" = 0; then
  python -u -m coreset_exp.acceptance > "$CORESET_OUT/independent_acceptance.log" 2>&1
  rc=$?
fi
printf '%s\n' "$rc" > "$CORESET_OUT/supervisor.exit"
exit "$rc"
