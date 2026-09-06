#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/data/libero_config"
export HF_HOME="$PWD/data/huggingface_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python}"
mkdir -p runs/corrected_baseline_diagnosis reports/corrected_baseline_diagnosis
exec 9>runs/corrected_baseline_diagnosis/queue.lock
flock -n 9
"$PYTHON" scripts/diagnose_corrected_baseline.py
