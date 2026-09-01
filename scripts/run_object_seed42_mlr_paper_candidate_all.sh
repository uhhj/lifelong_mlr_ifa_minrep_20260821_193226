#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

STATUS_DIR="$PROJECT_ROOT/runs/object_seed42_mlr_paper_candidate_driver"
mkdir -p "$STATUS_DIR"
printf 'TRAINING %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"

bash scripts/run_object_mlr_paper_candidate.sh > "$STATUS_DIR/train.log" 2>&1
printf 'EVALUATING %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"
bash scripts/eval_object_mlr_paper_candidate.sh > "$STATUS_DIR/eval.log" 2>&1

METRICS="$PROJECT_ROOT/reports/paper_reproduction_v2/seed42_mlr_candidate/metrics.json"
if "$PYTHON" - "$METRICS" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["promotion_gate"]["pass"] else 1)
PY
then
  printf 'COMPLETE_PROMOTED %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"
  if [[ "${AUTO_CONTINUE:-1}" == 1 ]]; then
    bash scripts/continue_paper_v2_after_seed42.sh > "$STATUS_DIR/continuation.log" 2>&1 &
  fi
else
  printf 'COMPLETE_NOT_PROMOTED %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"
fi
