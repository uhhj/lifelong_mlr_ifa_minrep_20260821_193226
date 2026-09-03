#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:?usage: $0 existing|pressure|test|smoke|bridge-seed|bridge-formal|evaluate|summarize|formal}"
REPORT_ROOT="$PROJECT_ROOT/reports/ifa_optimization_interaction"
DRIVER_ROOT="$PROJECT_ROOT/runs/ifa_optimization_interaction_driver"
mkdir -p "$REPORT_ROOT" "$DRIVER_ROOT"

bridge_seed() {
  local seed="${SEED:?SEED is required}"
  local sentinel="$REPORT_ROOT/bridge/seed${seed}.complete"
  if [[ -s "$sentinel" ]]; then
    echo "seed${seed} bridge already complete"
    return
  fi
  local run_dir="$PROJECT_ROOT/runs/object_seed${seed}_ifa_batchstep_gamma031_bridge/run_001"
  if [[ $(find "$run_dir" -maxdepth 1 -name 'mlrifa_*_model_ep*.pth' 2>/dev/null | wc -l) -ne 24 ]]; then
    bash scripts/run_object_ifa_batchstep_gamma031_bridge.sh "$seed"
  fi
  bash scripts/eval_object_ifa_batchstep_gamma031_bridge.sh "$seed"
  "$PYTHON" - "$PROJECT_ROOT/runs/object_seed${seed}_ifa_batchstep_gamma031_bridge/eval_seed${seed}.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
from compute_object_metrics import load_matrix
load_matrix(sys.argv[1])
PY
  mkdir -p "$REPORT_ROOT/bridge"
  printf 'complete %s\n' "$(date -Is)" > "$sentinel"
}

case "$MODE" in
  existing)
    "$PYTHON" scripts/summarize_ifa_optimization_interaction.py existing
    ;;
  pressure)
    "$PYTHON" scripts/diagnose_ifa_pressure_across_regimes.py
    ;;
  test)
    "$PYTHON" -m pytest -q tests/test_ifa_optimization_interaction.py
    ;;
  smoke)
    tmp=/tmp/ifa_optimization_interaction_smoke
    rm -rf -- "$tmp"
    mkdir -p "$tmp/run" "$tmp/buffer" "$tmp/log"
    PORT=$((30500 + RANDOM % 500))
    "$PYTHON" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend=c10d \
      --rdzv_endpoint="127.0.0.1:$PORT" libero/lifelong/lifelong_adaptation.py \
      seed=42 benchmark_name=LIBERO_OBJECT policy=BriefPolicy lifelong=mlr \
      folder="$DATA_ROOT" pretrain="$CHECKPOINT" train.n_epochs=100 \
      train.batch_size=10 train.num_workers=0 train.optimizer.kwargs.lr=1e-4 \
      +start_task=6 +end_task=7 +aug=false +feature_dir="$PROJECT_ROOT/features/object_seed42" \
      +buffer_dir="$tmp/buffer" +experiment_root="$tmp/run" +past_data_rate=0.5 \
      +ifa_enabled=true +ifa_weight=0.1 +gamma=0.31 +ifa_interaction_logging=true \
      +ifa_interaction_log_dir="$tmp/log" +repro_scheduler_mode=batch_step \
      +repro_arm=ifa_interaction_smoke +max_optimizer_updates=20 training=true
    "$PYTHON" - "$tmp/log/seed42_ifa_epoch_stats.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
assert len(rows) == 1 and int(rows[0]["optimizer_steps"]) == 20
assert all(math.isfinite(float(rows[0][key])) for key in ("mean_base_bc_loss", "mean_ifa_hinge", "ifa_to_bc_ratio"))
PY
    rm -rf -- "$tmp"
    ;;
  bridge-seed)
    bridge_seed
    ;;
  bridge-formal|formal)
    test ! -e "$DRIVER_ROOT/formal.lock" || { echo "formal bridge already running" >&2; exit 1; }
    mkdir "$DRIVER_ROOT/formal.lock"
    trap 'rmdir "$DRIVER_ROOT/formal.lock" 2>/dev/null || true' EXIT
    (CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 SEED=42 bash "$0" bridge-seed > "$DRIVER_ROOT/seed42.log" 2>&1) & p42=$!
    (CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 SEED=43 bash "$0" bridge-seed > "$DRIVER_ROOT/seed43.log" 2>&1) & p43=$!
    wait "$p42"; wait "$p43"
    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 SEED=44 bash "$0" bridge-seed > "$DRIVER_ROOT/seed44.log" 2>&1
    "$PYTHON" scripts/summarize_ifa_optimization_interaction.py bridge > "$DRIVER_ROOT/summary.log"
    printf 'COMPLETE %s\n' "$(date -Is)" > "$DRIVER_ROOT/status.txt"
    ;;
  evaluate)
    for seed in 42 43 44; do SEED="$seed" bridge_seed; done
    ;;
  summarize)
    "$PYTHON" scripts/summarize_ifa_optimization_interaction.py bridge
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
