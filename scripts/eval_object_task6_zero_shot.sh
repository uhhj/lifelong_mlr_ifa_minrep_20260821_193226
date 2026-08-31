#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

RUN_ROOT="$PROJECT_ROOT/runs/paper_v2_object_seed${SEED}_task6_zero_shot"
EPOCH_DIR="$RUN_ROOT/checkpoint"
REPORT_DIR="$PROJECT_ROOT/reports/paper_reproduction_v2/task6"
LOG_FILE="$REPORT_DIR/pretrain_task6_zero_shot.log"
JSON_FILE="$REPORT_DIR/pretrain_task6_zero_shot.json"
mkdir -p "$EPOCH_DIR" "$REPORT_DIR"
ln -sfn "$CHECKPOINT" "$EPOCH_DIR/mlrifa_6_model_ep99.pth"
: > "$LOG_FILE"

echo "[protocol] phase=zero_shot seed=$SEED gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=6 trials=20 checkpoint=$CHECKPOINT"
"$PYTHON" libero/lifelong/test.py \
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT" \
  eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=7 +aug=true \
  +epoch_dir="$EPOCH_DIR" +test_log_file="$LOG_FILE" +eval_current_only=true \
  '+eval_epochs=[99]' +experiment_root="$RUN_ROOT/eval" training=false

"$PYTHON" - "$LOG_FILE" "$JSON_FILE" "$CHECKPOINT" "$SEED" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"\[task 6 epoch 99 and 6 succ\.\]\s+([0-9.]+)", text)
if not match:
    raise SystemExit("zero-shot success record missing")
payload = {"benchmark": "LIBERO_OBJECT", "seed": int(sys.argv[4]), "task": 6,
           "trials": 20, "checkpoint": sys.argv[3], "success_rate": float(match.group(1))}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
