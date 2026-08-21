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

exec "$PYTHON" libero/lifelong/test.py \
  seed=42 \
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
  +end_task=10 \
  +aug=true \
  +eval_epochs='[70,75,80,85,90,95]' \
  +epoch_dir="$EPOCH_DIR" \
  +test_log_file="$LOG_FILE" \
  +experiment_root="$PROJECT_ROOT/runs/${RUN_NAME}_eval" \
  training=false
