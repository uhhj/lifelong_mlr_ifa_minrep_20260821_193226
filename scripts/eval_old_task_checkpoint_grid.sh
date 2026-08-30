#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

METHOD="${1:?usage: scripts/eval_old_task_checkpoint_grid.sh temporal_v1|zero_guard}"
case "$METHOD" in
  temporal_v1) RUN_NAME="${RUN_PREFIX}_mlr_temporal_igfa" ;;
  zero_guard) RUN_NAME="${RUN_PREFIX}_mlr_zero_directional_guard" ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

EPOCH_DIR="$PROJECT_ROOT/runs/$RUN_NAME/run_001"
LOG_FILE="${ALIGNMENT_LOG_FILE:-$PROJECT_ROOT/runs/$RUN_NAME/alignment_old_grid_seed${SEED}.log}"
EPOCH_LIST="${ALIGNMENT_EPOCHS:-[70,75,80,85,90,95]}"
SPECS="${ALIGNMENT_SPECS:-7:[6] 8:[6,7] 9:[6,7,8]}"
TRIALS="${ALIGNMENT_TRIALS:-20}"
test -d "$EPOCH_DIR" || { echo "Missing trained run: $EPOCH_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

echo "[protocol] phase=alignment-old-grid method=$METHOD seed=$SEED trials=$TRIALS epochs=$EPOCH_LIST specs=$SPECS no_training=true"
COMMON_ARGS=(
  seed="$SEED"
  benchmark_name=LIBERO_OBJECT
  policy=complete_policy
  lifelong=basic
  folder="$DATA_ROOT"
  bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files"
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files"
  pretrain="$CHECKPOINT"
  eval.n_eval="$TRIALS"
  eval.num_procs=20
  +start_task=6
  +end_task=10
  +aug=true
  +epoch_dir="$EPOCH_DIR"
  +test_log_file="$LOG_FILE"
  training=false
)

for SPEC in $SPECS; do
  IFS=: read -r STAGE TASK_IDS <<< "$SPEC"
  "$PYTHON" libero/lifelong/test.py "${COMMON_ARGS[@]}" \
    +eval_stage="$STAGE" \
    +eval_task_ids="$TASK_IDS" \
    +eval_epochs="$EPOCH_LIST" \
    +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_alignment_stage${STAGE}"
done
