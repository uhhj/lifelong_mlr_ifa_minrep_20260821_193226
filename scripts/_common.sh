#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config"
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"

PYTHON="${PYTHON:-$(command -v python)}"
CHECKPOINT="$PROJECT_ROOT/checkpoints/object_multitask_model_ep99.pth"
DATA_ROOT="$PROJECT_ROOT/data"
FEATURE_ROOT="$PROJECT_ROOT/features/object_seed42"

mkdir -p "$LIBERO_CONFIG_PATH" checkpoints data features buffers runs reports
printf '%s\n' \
  "benchmark_root: $PROJECT_ROOT/libero/libero" \
  "bddl_files: $PROJECT_ROOT/libero/libero/bddl_files" \
  "init_states: $PROJECT_ROOT/libero/libero/init_files" \
  "datasets: $DATA_ROOT" \
  "assets: $PROJECT_ROOT/libero/libero/assets" \
  > "$LIBERO_CONFIG_PATH/config_adaptation.yaml"

require_object_inputs() {
  test -f "$CHECKPOINT" || { echo "Missing official checkpoint: $CHECKPOINT" >&2; return 1; }
  test -d "$DATA_ROOT/libero_object" || { echo "Missing LIBERO-OBJECT data under $DATA_ROOT" >&2; return 1; }
}
