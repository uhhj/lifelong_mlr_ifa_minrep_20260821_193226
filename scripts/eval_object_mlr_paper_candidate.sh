#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

RUN_NAME="object_seed42_mlr_paper_candidate"
EPOCH_DIR="$PROJECT_ROOT/runs/$RUN_NAME/run_001"
LOG_FILE="$PROJECT_ROOT/runs/$RUN_NAME/eval_seed42.log"
test -d "$EPOCH_DIR" || { echo "Missing candidate training run: $EPOCH_DIR" >&2; exit 1; }
: > "$LOG_FILE"

echo "[protocol] phase=eval method=mlr_paper_candidate seed=42 gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=6-9 trials=20 checkpoints=70,75,80,85,90,95"

COMMON_ARGS=(
  seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files"
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT"
  eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true
  +epoch_dir="$EPOCH_DIR" +test_log_file="$LOG_FILE" training=false
)

"$PYTHON" libero/lifelong/test.py "${COMMON_ARGS[@]}" \
  +eval_current_only=true '+eval_epochs=[70,75,80,85,90,95]' \
  +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_eval_selection"

read -r E6 E7 E8 E9 < <("$PYTHON" - "$LOG_FILE" <<'PY'
import re, sys
pattern = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
records = {}
for line in open(sys.argv[1], encoding="utf-8"):
    match = pattern.search(line)
    if match:
        task, epoch, stage = map(int, match.group(1, 2, 3))
        records[(stage, epoch, task)] = float(match.group(4))
epochs = (70, 75, 80, 85, 90, 95)
print(*(max(epochs, key=lambda epoch: (records[(stage, epoch, stage)], epoch))
        for stage in range(6, 10)))
PY
)

for SPEC in "7:$E7:[6]" "8:$E8:[6,7]" "9:$E9:[6,7,8]"; do
  IFS=: read -r STAGE EPOCH TASKS <<< "$SPEC"
  "$PYTHON" libero/lifelong/test.py "${COMMON_ARGS[@]}" \
    +eval_stage="$STAGE" +eval_task_ids="$TASKS" +eval_epochs="[$EPOCH]" \
    +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_eval_stage${STAGE}"
done

"$PYTHON" scripts/compute_object_single_method_metrics.py \
  --log "$LOG_FILE" \
  --output-dir "$PROJECT_ROOT/reports/paper_reproduction_v2/seed42_mlr_candidate"
