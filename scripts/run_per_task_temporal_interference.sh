#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:-}"
REPORT_DIR="$PROJECT_ROOT/reports/per_task_temporal_interference_diagnosis"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$MODE" in
  smoke)
    SMOKE_DIR="$REPORT_DIR/smoke"
    "$PYTHON" scripts/diagnose_per_task_temporal_interference.py \
      --seeds 42 \
      --new-tasks 8 \
      --batches-per-stage 100 \
      --batch-size 10 \
      --checkpoint-epoch 95 \
      --checkpoint-mode entry \
      --output-dir "$SMOKE_DIR/raw" \
      --device cuda
    "$PYTHON" scripts/summarize_per_task_temporal_interference.py \
      --raw-dir "$SMOKE_DIR/raw" \
      --output-dir "$SMOKE_DIR"
    ;;
  formal)
    "$PYTHON" scripts/diagnose_per_task_temporal_interference.py \
      --seeds 42 43 44 \
      --new-tasks 7 8 9 \
      --batches-per-stage 400 \
      --batch-size 10 \
      --checkpoint-epoch 95 \
      --checkpoint-mode entry \
      --output-dir "$REPORT_DIR/raw" \
      --device cuda
    "$PYTHON" scripts/diagnose_per_task_temporal_interference.py \
      --seeds 44 \
      --new-tasks 8 9 \
      --batches-per-stage 400 \
      --batch-size 10 \
      --checkpoint-epoch 90 \
      --checkpoint-mode late \
      --output-dir "$REPORT_DIR/raw" \
      --device cuda
    "$PYTHON" scripts/summarize_per_task_temporal_interference.py \
      --raw-dir "$REPORT_DIR/raw" \
      --output-dir "$REPORT_DIR"
    ;;
  *)
    echo "usage: $0 {smoke|formal}" >&2
    exit 2
    ;;
esac
