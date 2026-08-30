#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:?usage: scripts/run_bc_rollout_retention_alignment.sh smoke|rollout|bc|summarize|formal}"
REPORT_DIR="$PROJECT_ROOT/reports/bc_rollout_retention_alignment"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_rollout_job() {
  local method="$1"
  local seed="$2"
  local gpu="$3"
  echo "[alignment-runner] rollout method=$method seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" SEED="$seed" \
    bash scripts/eval_old_task_checkpoint_grid.sh "$method"
}

run_rollout_grid() {
  (
    set -e
    for seed in 42 43 44; do
      run_rollout_job temporal_v1 "$seed" 0
    done
  ) &
  local lane0=$!
  (
    set -e
    for seed in 42 43 44; do
      run_rollout_job zero_guard "$seed" 1
    done
  ) &
  local lane1=$!
  local failed=0
  wait "$lane0" || failed=1
  wait "$lane1" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "One or more rollout lanes failed" >&2
    return 1
  fi
}

run_bc_grid() {
  mkdir -p "$REPORT_DIR"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    "$PYTHON" scripts/evaluate_bc_loss_checkpoint_grid.py \
      --root "$PROJECT_ROOT" \
      --output "$REPORT_DIR/bc_loss_checkpoint_grid.csv"
}

parse_rollout_grid() {
  mkdir -p "$REPORT_DIR"
  "$PYTHON" scripts/parse_rollout_checkpoint_grid.py \
    --root "$PROJECT_ROOT" \
    --output "$REPORT_DIR/rollout_checkpoint_grid.csv"
}

summarize_grid() {
  parse_rollout_grid
  "$PYTHON" scripts/summarize_bc_rollout_retention_alignment.py \
    --root "$PROJECT_ROOT" \
    --input-dir "$REPORT_DIR" \
    --output-dir "$REPORT_DIR"
}

run_smoke() {
  local smoke_root
  smoke_root=$(mktemp -d /tmp/bc_rollout_alignment_smoke.XXXXXX)
  ALIGNMENT_SMOKE_ROOT="$smoke_root"
  cleanup_smoke() {
    local resolved
    resolved=$(realpath -e "$ALIGNMENT_SMOKE_ROOT")
    case "$resolved" in
      /tmp/bc_rollout_alignment_smoke.*) rm -rf -- "$resolved" ;;
      *) echo "Unsafe smoke cleanup target: $resolved" >&2; return 1 ;;
    esac
  }
  trap cleanup_smoke EXIT
  for method in temporal_v1 zero_guard; do
    ALIGNMENT_LOG_FILE="$smoke_root/${method}.log" \
    ALIGNMENT_EPOCHS='[70,75]' \
    ALIGNMENT_SPECS='7:[6]' \
    ALIGNMENT_TRIALS=20 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}" \
    SEED=42 \
      bash scripts/eval_old_task_checkpoint_grid.sh "$method"
  done
  "$PYTHON" scripts/evaluate_bc_loss_checkpoint_grid.py \
    --root "$PROJECT_ROOT" \
    --methods temporal_v1 zero_guard \
    --seeds 42 --stages 7 --epochs 70 75 --tasks 6 \
    --output "$smoke_root/bc.csv"
  "$PYTHON" scripts/parse_rollout_checkpoint_grid.py \
    --root "$PROJECT_ROOT" \
    --methods temporal_v1 zero_guard --seeds 42 --epochs 70 75 \
    --old-only \
    --old-log-template "$smoke_root/{method}.log" \
    --output "$smoke_root/rollout.csv"
  "$PYTHON" - "$smoke_root/bc.csv" "$smoke_root/rollout.csv" <<'PY'
import csv
import math
import sys
from libero.lifelong.analysis.bc_rollout_retention_alignment import (
    alignment_metrics,
    matched_method_contrasts,
)
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    bc = list(csv.DictReader(handle))
with open(sys.argv[2], newline="", encoding="utf-8") as handle:
    rollout = list(csv.DictReader(handle))
rows = matched_method_contrasts(bc, rollout)
assert len(rows) == 2, len(rows)
assert all(math.isfinite(float(row["relative_bc_advantage"])) for row in rows)
assert all(math.isfinite(float(row["rollout_advantage_pp"])) for row in rows)
summary = alignment_metrics(rows)
print(f"[alignment-smoke] PASS matched={len(rows)} quadrants={[row['quadrant'] for row in rows]}")
PY
}

case "$MODE" in
  smoke) run_smoke ;;
  rollout) run_rollout_grid ;;
  bc) run_bc_grid ;;
  summarize) summarize_grid ;;
  formal)
    echo "[protocol] phase=formal-alignment no_training=true methods=2 seeds=3 epochs=6 trials=20"
    run_rollout_grid
    run_bc_grid
    summarize_grid
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
