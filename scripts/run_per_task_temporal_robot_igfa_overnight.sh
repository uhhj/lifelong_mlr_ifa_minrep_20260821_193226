#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate /home/dm/miniconda3/envs/wzp_mlr_tmp

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"

QUEUE_DIR="$PROJECT_ROOT/runs/per_task_temporal_v2_queue"
mkdir -p "$QUEUE_DIR"
printf 'RUNNING %s\n' "$(date --iso-8601=seconds)" > "$QUEUE_DIR/status.txt"
trap 'printf "FAILED line=%s time=%s\n" "$LINENO" "$(date --iso-8601=seconds)" > "$QUEUE_DIR/status.txt"' ERR

for seed in 42 43 44; do
  printf 'TRAIN seed=%s time=%s\n' "$seed" "$(date --iso-8601=seconds)" | tee -a "$QUEUE_DIR/master.log"
  SEED="$seed" bash scripts/run_object_mlr_per_task_temporal_robot_igfa.sh \
    2>&1 | tee "$QUEUE_DIR/seed${seed}_train.log"
  printf 'EVAL seed=%s time=%s\n' "$seed" "$(date --iso-8601=seconds)" | tee -a "$QUEUE_DIR/master.log"
  SEED="$seed" bash test.sbatch mlr_per_task_temporal_igfa \
    2>&1 | tee "$QUEUE_DIR/seed${seed}_eval_console.log"
done

python scripts/compute_per_task_temporal_robot_igfa_metrics.py \
  2>&1 | tee "$QUEUE_DIR/metrics.log"
printf 'COMPLETE %s\n' "$(date --iso-8601=seconds)" > "$QUEUE_DIR/status.txt"
