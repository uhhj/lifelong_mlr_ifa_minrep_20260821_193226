#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate /home/dm/miniconda3/envs/wzp_mlr_tmp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config"

MODE="${1:?usage: scripts/run_finite_step_functional_probe.sh smoke|formal}"
case "$MODE" in
  smoke)
    SMOKE_DIR="$(mktemp -d /tmp/finite_step_functional_probe_smoke.XXXXXX)"
    trap 'rm -rf -- "$SMOKE_DIR"' EXIT
    python scripts/diagnose_finite_step_functional_probe.py smoke \
      --output-dir "$SMOKE_DIR"
    ;;
  formal)
    REPORT_DIR="$PROJECT_ROOT/reports/finite_step_functional_probe_validity"
    mkdir -p "$REPORT_DIR"
    python scripts/diagnose_finite_step_functional_probe.py formal \
      --output-dir "$REPORT_DIR"
    python scripts/summarize_finite_step_functional_probe.py \
      --report-dir "$REPORT_DIR"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
