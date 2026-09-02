#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DRIVER_ROOT="$PROJECT_ROOT/runs/paper_v2_auto_continue_driver"
LOCK_DIR="$PROJECT_ROOT/runs/paper_v2_auto_continue.lock"
SEED42_STATUS="$PROJECT_ROOT/runs/object_seed42_mlr_paper_candidate_driver/status.txt"
MLR_REPORT="$PROJECT_ROOT/reports/paper_reproduction_v2/mlr_formal"
RESUME_FROM_MLR_SUMMARY="${RESUME_FROM_MLR_SUMMARY:-0}"
ALLOW_DOCUMENTED_MLR_DELTA="${ALLOW_DOCUMENTED_MLR_DELTA:-0}"
mkdir -p "$DRIVER_ROOT"

if [[ "$RESUME_FROM_MLR_SUMMARY" != 1 ]]; then
  printf 'WAITING_SEED42 %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
  while ! grep -q '^COMPLETE_' "$SEED42_STATUS" 2>/dev/null; do sleep 60; done
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another paper-v2 continuation owns $LOCK_DIR" >&2
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
if ! grep -q '^COMPLETE_PROMOTED' "$SEED42_STATUS"; then
  printf 'STOPPED_SEED42_GATE %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
  exit 0
fi

if [[ "$RESUME_FROM_MLR_SUMMARY" == 1 ]]; then
  test -s "$MLR_REPORT/three_seed_summary.json"
  printf 'RESUMING_FROM_MLR_SUMMARY %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
else
  for SEED in 43 44; do
    printf 'MLR_TRAIN_SEED%s %s\n' "$SEED" "$(date -Is)" > "$DRIVER_ROOT/status.txt"
    bash scripts/run_object_paper_v2_method.sh mlr "$SEED" > "$DRIVER_ROOT/mlr_seed${SEED}_train.log" 2>&1
    printf 'MLR_EVAL_SEED%s %s\n' "$SEED" "$(date -Is)" > "$DRIVER_ROOT/status.txt"
    bash scripts/eval_object_paper_v2_method.sh mlr "$SEED" > "$DRIVER_ROOT/mlr_seed${SEED}_eval.log" 2>&1
  done

  "$PYTHON" scripts/compute_paper_v2_method_multiseed.py --method mlr --output-dir "$MLR_REPORT" > "$DRIVER_ROOT/mlr_summary.log"
fi
if ! "$PYTHON" - "$MLR_REPORT/three_seed_summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["pass_next_gate"] else 1)
PY
then
  if [[ "$ALLOW_DOCUMENTED_MLR_DELTA" == 1 ]]; then
    printf 'MLR_GATE_OVERRIDDEN_DOCUMENTED_DELTA %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
  else
    printf 'STOPPED_MLR_GATE %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
    exit 0
  fi
fi

for SEED in 42 43 44; do
  printf 'IFA_TRAIN_SEED%s %s\n' "$SEED" "$(date -Is)" > "$DRIVER_ROOT/status.txt"
  bash scripts/run_object_paper_v2_method.sh mlr_ifa "$SEED" > "$DRIVER_ROOT/ifa_seed${SEED}_train.log" 2>&1
  printf 'IFA_EVAL_SEED%s %s\n' "$SEED" "$(date -Is)" > "$DRIVER_ROOT/status.txt"
  bash scripts/eval_object_paper_v2_method.sh mlr_ifa "$SEED" > "$DRIVER_ROOT/ifa_seed${SEED}_eval.log" 2>&1
done

"$PYTHON" scripts/compute_paper_v2_method_multiseed.py --method mlr_ifa \
  --output-dir "$PROJECT_ROOT/reports/paper_reproduction_v2/ifa_formal" > "$DRIVER_ROOT/ifa_summary.log"
printf 'COMPLETE %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
