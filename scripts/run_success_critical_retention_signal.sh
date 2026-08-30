#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:?usage: scripts/run_success_critical_retention_signal.sh smoke|evaluate|summarize|formal}"
REPORT_DIR="$PROJECT_ROOT/reports/success_critical_retention_signal"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

evaluate_formal() {
  mkdir -p "$REPORT_DIR"
  "$PYTHON" scripts/evaluate_policy_distribution_retention.py \
    --root "$PROJECT_ROOT" \
    --build-phase-manifest \
    --phase-manifest "$REPORT_DIR/phase_probe_manifest.csv" \
    --output "$REPORT_DIR/signal_checkpoint_grid.csv" \
    --device cuda
}

summarize_formal() {
  "$PYTHON" scripts/summarize_policy_distribution_retention.py \
    --root "$PROJECT_ROOT" \
    --input-dir "$REPORT_DIR" \
    --output-dir "$REPORT_DIR"
}

run_smoke() {
  SUCCESS_CRITICAL_SMOKE_ROOT=$(mktemp -d /tmp/success_critical_retention_smoke.XXXXXX)
  cleanup_smoke() {
    local resolved
    resolved=$(realpath -e "$SUCCESS_CRITICAL_SMOKE_ROOT")
    case "$resolved" in
      /tmp/success_critical_retention_smoke.*) rm -rf -- "$resolved" ;;
      *) echo "Unsafe smoke cleanup target: $resolved" >&2; return 1 ;;
    esac
  }
  trap cleanup_smoke EXIT
  "$PYTHON" scripts/evaluate_policy_distribution_retention.py \
    --root "$PROJECT_ROOT" \
    --methods temporal_v1 zero_guard \
    --seeds 42 --tasks 6 --stages 7 --epochs 70 75 \
    --samples-per-bin 8 --global-limit 32 --mc-samples 8 \
    --build-phase-manifest \
    --phase-manifest "$SUCCESS_CRITICAL_SMOKE_ROOT/phase.csv" \
    --output "$SUCCESS_CRITICAL_SMOKE_ROOT/grid.csv" \
    --device cuda
  "$PYTHON" - "$SUCCESS_CRITICAL_SMOKE_ROOT/grid.csv" \
    "$PROJECT_ROOT/reports/bc_rollout_retention_alignment/rollout_checkpoint_grid.csv" <<'PY'
import csv
import math
import sys
from libero.lifelong.analysis.policy_distribution_retention import (
    matched_signal_contrasts,
    signal_alignment_metrics,
)
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    grid = list(csv.DictReader(handle))
with open(sys.argv[2], newline="", encoding="utf-8") as handle:
    rollout = [
        row for row in csv.DictReader(handle)
        if int(row["seed"]) == 42 and int(row["stage"]) == 7
        and int(row["epoch"]) in (70, 75) and int(row["task"]) == 6
    ]
assert len(grid) == 4 and len(rollout) == 4
numeric = [
    "bc_global_mean", "bc_global_q90", "rollout_kl_global_mean",
    "rollout_w2_global_mean", "full_gmm_fwdkl_global_mean",
    "rollout_kl_phase_worst_bin", "rollout_w2_phase_late_bin",
]
assert all(math.isfinite(float(row[name])) for row in grid for name in numeric)
matched = matched_signal_contrasts(
    grid, rollout, "rollout_kl_global_mean", "primary_policy"
)
assert len(matched) == 2
signal_alignment_metrics(matched)
print(f"[success-critical-smoke] PASS grid={len(grid)} matched={len(matched)}")
PY
}

case "$MODE" in
  smoke) run_smoke ;;
  evaluate) evaluate_formal ;;
  summarize) summarize_formal ;;
  formal)
    echo "[protocol] H19 formal no_training=true no_optimizer_step=true no_rollout=true"
    evaluate_formal
    summarize_formal
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
