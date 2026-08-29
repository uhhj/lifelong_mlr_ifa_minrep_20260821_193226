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

MODE="${1:?usage: scripts/run_first_order_retention_validity.sh smoke|formal}"
case "$MODE" in
  smoke)
    SMOKE_DIR="$(mktemp -d /tmp/first_order_retention_validity_smoke.XXXXXX)"
    trap 'rm -rf -- "$SMOKE_DIR"' EXIT
    python scripts/diagnose_first_order_retention_validity.py layer_a \
      --methods V2 --seeds 42 --stages 8 --batches 30 \
      --output-dir "$SMOKE_DIR"
    python scripts/diagnose_first_order_retention_validity.py checkpoint \
      --methods V2 --seeds 42 --stages 8 --short-epochs 90 95 \
      --probe-size 64 --skip-long --output-dir "$SMOKE_DIR"
    python - "$SMOKE_DIR" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
layer = list(csv.DictReader((root / "layer_a_raw.csv").open()))
checkpoint = list(csv.DictReader((root / "checkpoint_transition_raw.csv").open()))
assert layer and len(checkpoint) == 2
for row in layer + checkpoint:
    for value in row.values():
        if value not in ("", None):
            try:
                assert math.isfinite(float(value))
            except ValueError:
                pass
print(f"[smoke-pass] layer_relations={len(layer)} checkpoint_cases={len(checkpoint)}")
PY
    ;;
  formal)
    REPORT_DIR="$PROJECT_ROOT/reports/first_order_retention_validity"
    mkdir -p "$REPORT_DIR"
    python scripts/diagnose_first_order_retention_validity.py layer_a \
      --methods V1 V2 --seeds 42 43 44 --stages 7 8 9 \
      --stage7-batches 50 --batches 150 --batch-size 10 \
      --diagnostic-seed 20260830 --output-dir "$REPORT_DIR"
    python scripts/diagnose_first_order_retention_validity.py checkpoint \
      --methods V1 V2 --seeds 42 43 44 --stages 7 8 9 \
      --short-epochs 70 75 80 85 90 95 --probe-size 256 \
      --probe-seed 20260830 --probe-batch-size 32 \
      --output-dir "$REPORT_DIR"
    python scripts/summarize_first_order_retention_validity.py \
      --report-dir "$REPORT_DIR"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
