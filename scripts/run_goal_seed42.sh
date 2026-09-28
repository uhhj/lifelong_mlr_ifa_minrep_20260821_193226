#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
printf 'START %s\n' "$(date -Is)"
trap 'rc=$?; printf "END %s EXIT %s\n" "$(date -Is)" "$rc"' EXIT
python "$root/scripts/goal_v2_driver.py" "$@"
