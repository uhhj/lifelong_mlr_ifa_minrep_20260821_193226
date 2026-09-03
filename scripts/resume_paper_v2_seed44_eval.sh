#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DRIVER_ROOT="$PROJECT_ROOT/runs/paper_v2_auto_continue_driver"
STATUS_FILE="$DRIVER_ROOT/status.txt"
mkdir -p "$DRIVER_ROOT"

test -s "$PROJECT_ROOT/runs/object_seed42_mlr_ifa_paper_v2/eval_seed42.log"
test -s "$PROJECT_ROOT/runs/object_seed43_mlr_ifa_paper_v2/eval_seed43.log"
test -d "$PROJECT_ROOT/runs/object_seed44_mlr_ifa_paper_v2/run_001"

printf 'IFA_EVAL_SEED44_RESTARTED %s\n' "$(date -Is)" > "$STATUS_FILE"
bash scripts/eval_object_paper_v2_method.sh mlr_ifa 44 \
  > "$DRIVER_ROOT/ifa_seed44_eval.log" 2>&1

printf 'IFA_SUMMARIZING %s\n' "$(date -Is)" > "$STATUS_FILE"
"$PYTHON" scripts/compute_paper_v2_method_multiseed.py --method mlr_ifa \
  --output-dir "$PROJECT_ROOT/reports/paper_reproduction_v2/ifa_formal" \
  > "$DRIVER_ROOT/ifa_summary.log"
printf 'COMPLETE %s\n' "$(date -Is)" > "$STATUS_FILE"
