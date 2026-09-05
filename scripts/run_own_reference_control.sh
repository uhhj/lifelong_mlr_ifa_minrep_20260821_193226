#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs
DRIVER="$PROJECT_ROOT/runs/own_reference_control_driver"
mkdir -p "$DRIVER"
exec 9>"$DRIVER/queue.lock"
flock -n 9 || { echo 'Queue already active'; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "FAILED rc=%s %s\n" "$rc" "$(date -Is)" > "$DRIVER/status.txt"; fi' EXIT
for SEED in 42 43 44; do
  RUN_DIR="$PROJECT_ROOT/runs/own_reference_control/ifa/seed${SEED}"
  BUFFER_DIR="$PROJECT_ROOT/buffers/own_reference_control/ifa/seed${SEED}"
  FEATURE_ROOT="$PROJECT_ROOT/features/object_seed${SEED}"
  mkdir -p "$RUN_DIR" "$BUFFER_DIR"
  if [[ ! -s "$RUN_DIR/TRAINING_COMPLETE" ]]; then
    [[ ! -d "$RUN_DIR/run_001" ]] || { echo "Incomplete run needs inspection: $RUN_DIR"; exit 1; }
    echo "TRAIN_SEED${SEED} $(date -Is)" > "$DRIVER/status.txt"
    echo "[protocol] seed=$SEED GPU=$CUDA_VISIBLE_DEVICES tasks=6-9 epochs=100 batch=10 AdamW lr=1e-4 gamma=.30 lambda=.1 own_reference_fix=true feature=$FEATURE_ROOT checkpoint=$CHECKPOINT"
    "$PYTHON" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:31842" \
      libero/lifelong/lifelong_adaptation.py \
      seed="$SEED" benchmark_name=LIBERO_OBJECT policy=BriefPolicy lifelong=mlr \
      folder="$DATA_ROOT" pretrain="$CHECKPOINT" \
      train.n_epochs=100 train.batch_size=10 train.num_workers=0 \
      train.optimizer.kwargs.lr=1e-4 train.optimizer.kwargs.betas='[0.9,0.999]' \
      train.optimizer.kwargs.eps=1e-8 train.optimizer.kwargs.weight_decay=0.1 \
      +start_task=6 +end_task=10 +aug=false +feature_dir="$FEATURE_ROOT" +buffer_dir="$BUFFER_DIR" \
      +experiment_root="$RUN_DIR" +past_data_rate=0.5 +ifa_enabled=true +ifa_weight=0.1 +gamma=0.30 \
      +ifa_own_reference_fix=true +repro_scheduler_mode=batch_step +repro_arm=own_reference_control \
      +save_final_checkpoint=true training=true > "$DRIVER/train_seed${SEED}.log" 2>&1
    "$PYTHON" scripts/validate_own_reference_control.py "$SEED" > "$DRIVER/validate_seed${SEED}.log" 2>&1
    date -Is > "$RUN_DIR/TRAINING_COMPLETE"
  fi
  if [[ ! -s "$RUN_DIR/PRIMARY_EVAL_COMPLETE" ]]; then
    echo "EVAL_SEED${SEED} $(date -Is)" > "$DRIVER/status.txt"
    [[ ! -e "$RUN_DIR/primary_ep99.log" ]] || { echo 'Partial evaluation needs inspection'; exit 1; }
    "$PYTHON" libero/lifelong/test.py \
      seed="$SEED" benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
      folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
      init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT" \
      eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true \
      +epoch_dir="$RUN_DIR/run_001" +test_log_file="$RUN_DIR/primary_ep99.log" \
      '+eval_epochs=[99]' +experiment_root="$RUN_DIR/eval_primary_ep99" training=false \
      > "$DRIVER/eval_seed${SEED}.log" 2>&1
    "$PYTHON" - "$RUN_DIR/primary_ep99.log" <<'PY'
import sys
sys.path.insert(0, 'scripts')
from summarize_final_paper_text_faithful import fixed_epoch_matrix
fixed_epoch_matrix(sys.argv[1])
PY
    date -Is > "$RUN_DIR/PRIMARY_EVAL_COMPLETE"
  fi
done
echo "COMPLETE $(date -Is)" > "$DRIVER/status.txt"
