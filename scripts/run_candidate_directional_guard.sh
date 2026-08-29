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

MODE="${1:?usage: scripts/run_candidate_directional_guard.sh smoke|formal|summarize}"
case "$MODE" in
  smoke)
    SMOKE_ROOT="$(mktemp -d /tmp/candidate_directional_guard_smoke.XXXXXX)"
    trap 'rm -rf -- "$SMOKE_ROOT"' EXIT
    python scripts/diagnose_candidate_directional_guard.py \
      --seed 42 --stage 8 --warmup-epochs 2 --intervention-updates 20 \
      --guard-scale 0.5 --label-probe-size 64 --score-probe-size 16 \
      --current-probe-size 64 --probe-batch-size 16 --batch-size 10 \
      --device cuda --output-dir "$SMOKE_ROOT/raw"
    python - "$SMOKE_ROOT/raw" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = list(csv.DictReader((root / "seed42_stage8_steps.csv").open()))
probes = list(csv.DictReader((root / "seed42_stage8_probe.csv").open()))
assert len(steps) == 40
assert {(row["arm"], int(row["update"])) for row in probes} == {
    (arm, update) for arm in ("control", "guard") for update in (0, 20)
}
assert all(math.isfinite(float(row["candidate_step_cosine"])) for row in steps)
control = next(row for row in probes if row["arm"] == "control" and row["update"] == "0")
guard = next(row for row in probes if row["arm"] == "guard" and row["update"] == "0")
assert abs(float(control["old_mean_loss"]) - float(guard["old_mean_loss"])) <= 1e-6
assert abs(float(control["current_loss"]) - float(guard["current_loss"])) <= 1e-6
print("[smoke-pass] arms=2 updates_per_arm=20 branch_parity=PASS")
PY
    ;;
  formal)
    REPORT_ROOT="$PROJECT_ROOT/reports/candidate_directional_guard_smoke"
    mkdir -p "$REPORT_ROOT/raw"
    for seed in 42 43; do
      for stage in 8 9; do
        rm -f -- \
          "$REPORT_ROOT/raw/seed${seed}_stage${stage}_steps.csv" \
          "$REPORT_ROOT/raw/seed${seed}_stage${stage}_probe.csv" \
          "$REPORT_ROOT/raw/seed${seed}_stage${stage}_old_task_probe.csv"
        python scripts/diagnose_candidate_directional_guard.py \
          --seed "$seed" --stage "$stage" --warmup-epochs 10 \
          --intervention-updates 300 --guard-scale 0.5 \
          --label-probe-size 256 --score-probe-size 64 \
          --current-probe-size 256 --probe-batch-size 32 \
          --batch-size 10 --device cuda --output-dir "$REPORT_ROOT/raw"
      done
    done
    python scripts/summarize_candidate_directional_guard.py \
      --report-dir "$REPORT_ROOT"
    ;;
  summarize)
    python scripts/summarize_candidate_directional_guard.py \
      --report-dir "$PROJECT_ROOT/reports/candidate_directional_guard_smoke"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
