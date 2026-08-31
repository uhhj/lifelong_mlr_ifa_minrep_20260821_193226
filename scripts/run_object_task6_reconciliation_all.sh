#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

STATUS_DIR="$PROJECT_ROOT/runs/paper_v2_task6_driver"
mkdir -p "$STATUS_DIR"
printf 'RUNNING %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"

run_on_gpu() {
  local gpu="$1" log="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$@" > "$STATUS_DIR/$log" 2>&1
}

(
  run_on_gpu 0 zero_shot.log bash scripts/eval_object_task6_zero_shot.sh
  run_on_gpu 0 train_release18_legacy.log bash scripts/run_object_task6_reconciliation.sh release18_legacy
  run_on_gpu 0 train_batch10_linear_step.log bash scripts/run_object_task6_reconciliation.sh batch10_linear_step
) &
GPU0_PID=$!

(
  run_on_gpu 1 train_current_legacy.log bash scripts/run_object_task6_reconciliation.sh current_legacy
  run_on_gpu 1 train_batch10_constant.log bash scripts/run_object_task6_reconciliation.sh batch10_constant
) &
GPU1_PID=$!

wait "$GPU0_PID"
wait "$GPU1_PID"
printf 'TRAINING_COMPLETE %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"

(
  run_on_gpu 0 eval_current_legacy.log bash scripts/eval_object_task6_reconciliation.sh current_legacy
  run_on_gpu 0 eval_batch10_linear_step.log bash scripts/eval_object_task6_reconciliation.sh batch10_linear_step
) &
GPU0_PID=$!

(
  run_on_gpu 1 eval_release18_legacy.log bash scripts/eval_object_task6_reconciliation.sh release18_legacy
  run_on_gpu 1 eval_batch10_constant.log bash scripts/eval_object_task6_reconciliation.sh batch10_constant
) &
GPU1_PID=$!

wait "$GPU0_PID"
wait "$GPU1_PID"
"$PYTHON" scripts/summarize_object_task6_reconciliation.py > "$STATUS_DIR/verdict.txt"
printf 'COMPLETE %s\n' "$(date -Is)" > "$STATUS_DIR/status.txt"
