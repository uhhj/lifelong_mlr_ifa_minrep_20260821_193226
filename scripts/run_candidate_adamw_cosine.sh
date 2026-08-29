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

MODE="${1:?usage: scripts/run_candidate_adamw_cosine.sh smoke|formal|summarize}"
case "$MODE" in
  smoke)
    SMOKE_ROOT="$(mktemp -d /tmp/candidate_adamw_cosine_smoke.XXXXXX)"
    trap 'rm -rf -- "$SMOKE_ROOT"' EXIT
    python scripts/diagnose_candidate_adamw_cosine.py \
      --seed 42 --stage 8 --epochs 3 --burnin-epochs 2 \
      --anchor-interval 20 --label-probe-size 64 --score-probe-size 16 \
      --probe-batch-size 16 --batch-size 10 --device cuda \
      --output-dir "$SMOKE_ROOT/raw"
    python - "$SMOKE_ROOT/raw/seed42_stage8_anchor.csv" <<'PY'
import csv
import math
import sys
from libero.lifelong.analysis.online_first_order_margin import build_future_cases

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
future = build_future_cases(rows, 20)
assert len(rows) >= 2 and len(future) >= 1
for row in rows:
    for field in (
        "candidate_actual_delta_cosine", "relative_l2_error",
        "risk_cosine_abs_error", "candidate_step_cosine", "actual_step_cosine",
    ):
        assert math.isfinite(float(row[field]))
print(f"[smoke-pass] anchors={len(rows)} h20_cases={len(future)}")
PY
    ;;
  formal)
    REPORT_ROOT="$PROJECT_ROOT/reports/candidate_adamw_step_cosine"
    mkdir -p "$REPORT_ROOT/raw"
    for seed in 42 43 44; do
      for stage in 8 9; do
        python scripts/diagnose_candidate_adamw_cosine.py \
          --seed "$seed" --stage "$stage" --epochs 15 --burnin-epochs 2 \
          --anchor-interval 100 --label-probe-size 256 --score-probe-size 64 \
          --probe-batch-size 32 --batch-size 10 --device cuda \
          --output-dir "$REPORT_ROOT/raw"
      done
    done
    python scripts/summarize_candidate_adamw_cosine.py \
      --report-dir "$REPORT_ROOT"
    ;;
  summarize)
    python scripts/summarize_candidate_adamw_cosine.py \
      --report-dir "$PROJECT_ROOT/reports/candidate_adamw_step_cosine"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
