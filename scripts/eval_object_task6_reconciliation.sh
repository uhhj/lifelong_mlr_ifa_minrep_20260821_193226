#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

ARM="${1:?usage: scripts/eval_object_task6_reconciliation.sh ARM}"
RUN_TAG="paper_v2_object_seed${SEED}_task6_${ARM}"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_TAG"
EPOCH_DIR="$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'run_*' | sort | tail -1)"
test -n "$EPOCH_DIR" || { echo "Missing trained task6 run: $RUN_ROOT" >&2; exit 1; }

REPORT_DIR="$PROJECT_ROOT/reports/paper_reproduction_v2/task6/$ARM"
LOG_FILE="$REPORT_DIR/task6_eval.log"
CSV_FILE="$REPORT_DIR/task6_eval.csv"
mkdir -p "$REPORT_DIR"
: > "$LOG_FILE"

EPOCHS='[9,19,29,39,49,59,69,70,75,79,80,85,89,90,95,99]'
echo "[protocol] phase=eval seed=$SEED arm=$ARM gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=6 trials=20 epochs=$EPOCHS checkpoint_dir=$EPOCH_DIR"

"$PYTHON" libero/lifelong/test.py \
  seed="$SEED" \
  benchmark_name=LIBERO_OBJECT \
  policy=complete_policy \
  lifelong=basic \
  folder="$DATA_ROOT" \
  bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" \
  pretrain="$CHECKPOINT" \
  eval.n_eval=20 \
  eval.num_procs=20 \
  +start_task=6 \
  +end_task=7 \
  +aug=true \
  +epoch_dir="$EPOCH_DIR" \
  +test_log_file="$LOG_FILE" \
  +eval_current_only=true \
  +eval_epochs="$EPOCHS" \
  +experiment_root="$PROJECT_ROOT/runs/${RUN_TAG}_eval" \
  training=false

"$PYTHON" - "$LOG_FILE" "$CSV_FILE" "$ARM" "$SEED" <<'PY'
import csv
import re
import sys

pattern = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    match = pattern.search(line)
    if match:
        task, epoch, stage = map(int, match.group(1, 2, 3))
        rows.append({"seed": int(sys.argv[4]), "arm": sys.argv[3], "stage": stage,
                     "task": task, "epoch": epoch, "trials": 20,
                     "success_rate": float(match.group(4))})
with open(sys.argv[2], "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY
