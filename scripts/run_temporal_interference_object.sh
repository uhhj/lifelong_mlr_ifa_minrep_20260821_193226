#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

MODE="${1:-}"
OUTPUT="reports/temporal_interference_diagnosis"
mkdir -p "$OUTPUT/logs"

case "$MODE" in
  smoke)
    rm -rf "$OUTPUT/smoke"
    mkdir -p "$OUTPUT/smoke/logs"
    python scripts/diagnose_temporal_interference.py \
      --method mlr --seeds 42 --new-tasks 7 --pairs 6-7 \
      --samples-per-bin 8 --gradient-batch-size 8 \
      --output-dir "$OUTPUT/smoke" \
      2>&1 | tee "$OUTPUT/smoke/logs/smoke.log"
    python scripts/summarize_temporal_interference.py \
      --input-dir "$OUTPUT/smoke/raw" --output-dir "$OUTPUT/smoke"
    ;;
  formal)
    rm -rf "$OUTPUT/raw" "$OUTPUT/figures"
    rm -f "$OUTPUT/alignment_summary.csv" "$OUTPUT/task_level_pairs.csv" \
      "$OUTPUT/temporal_pair_summary.csv" "$OUTPUT/temporal_3seed_summary.csv"
    python scripts/diagnose_temporal_interference.py \
      --method mlr --seeds 42 43 44 --new-tasks 7 8 9 \
      --samples-per-bin 32 --gradient-batch-size 8 \
      --output-dir "$OUTPUT" \
      2>&1 | tee "$OUTPUT/logs/formal_mlr.log"
    python scripts/summarize_temporal_interference.py \
      --input-dir "$OUTPUT/raw" --output-dir "$OUTPUT"
    ;;
  ifa)
    python scripts/diagnose_temporal_interference.py \
      --method mlr_ifa --seeds 42 43 44 --new-tasks 7 8 9 \
      --pairs 6-7 6-8 7-8 7-9 \
      --samples-per-bin 32 --gradient-batch-size 8 --append \
      --output-dir "$OUTPUT" \
      2>&1 | tee "$OUTPUT/logs/secondary_mlr_ifa.log"
    python scripts/summarize_temporal_interference.py \
      --input-dir "$OUTPUT/raw" --output-dir "$OUTPUT"
    ;;
  *)
    echo "usage: $0 {smoke|formal|ifa}" >&2
    exit 2
    ;;
esac
