#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

METHOD="${1:?usage: scripts/eval_object.sh mlr|mlr_ifa}"
case "$METHOD" in
  mlr) RUN_NAME=object_seed42_mlr ;;
  mlr_ifa) RUN_NAME=object_seed42_mlr_ifa ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

EPOCH_DIR="$PROJECT_ROOT/runs/$RUN_NAME/run_001"
LOG_FILE="$PROJECT_ROOT/runs/$RUN_NAME/eval_seed42.log"
test -d "$EPOCH_DIR" || { echo "Missing trained run: $EPOCH_DIR" >&2; exit 1; }
: > "$LOG_FILE"

COMMON_ARGS=(
  seed=42
  benchmark_name=LIBERO_OBJECT
  policy=complete_policy
  lifelong=basic
  folder="$DATA_ROOT"
  bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files"
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files"
  pretrain="$CHECKPOINT"
  eval.n_eval=20
  eval.num_procs=20
  +start_task=6
  +end_task=10
  +aug=true
  +epoch_dir="$EPOCH_DIR"
  +test_log_file="$LOG_FILE"
  training=false
)

# Phase 1: evaluate only the current task at every released checkpoint.
"$PYTHON" libero/lifelong/test.py "${COMMON_ARGS[@]}" \
  +eval_current_only=true \
  +eval_epochs='[70,75,80,85,90,95]' \
  +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_eval_selection"

read -r E6 E7 E8 E9 < <("$PYTHON" - "$LOG_FILE" <<'PY'
import re
import sys

line = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
records = {}
for text in open(sys.argv[1], encoding="utf-8"):
    match = line.search(text)
    if match:
        task, epoch, stage = map(int, match.group(1, 2, 3))
        records[(stage, epoch, task)] = float(match.group(4))
epochs = (70, 75, 80, 85, 90, 95)
print(*(max(epochs, key=lambda epoch: (records[(stage, epoch, stage)], epoch))
        for stage in range(6, 10)))
PY
)

# Phase 2: evaluate old tasks only at each stage's selected checkpoint.
for SPEC in "7:$E7:[6]" "8:$E8:[6,7]" "9:$E9:[6,7,8]"; do
  IFS=: read -r STAGE EPOCH TASKS <<< "$SPEC"
  "$PYTHON" libero/lifelong/test.py "${COMMON_ARGS[@]}" \
    +eval_stage="$STAGE" \
    +eval_task_ids="$TASKS" \
    +eval_epochs="[$EPOCH]" \
    +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_eval_stage${STAGE}"
done

