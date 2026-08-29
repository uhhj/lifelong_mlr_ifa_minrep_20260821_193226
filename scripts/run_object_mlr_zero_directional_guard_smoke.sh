#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs
test -d "$FEATURE_ROOT/7" || {
  echo "Missing seed42 latent features: $FEATURE_ROOT" >&2
  exit 1
}

TEMPORAL_BIN_MAP="$PROJECT_ROOT/metadata/temporal_bins/object_seed${SEED}.json"
test -f "$TEMPORAL_BIN_MAP" || {
  echo "Missing frozen temporal sidecar: $TEMPORAL_BIN_MAP" >&2
  exit 1
}

SMOKE_NAME="object_seed${SEED}_mlr_zero_directional_guard_smoke"
echo "[protocol] phase=smoke method=mlr_zero_directional_guard seed=$SEED tasks=6-7 epochs=2 batch_size=10 bins=5 temporal_gate_strength=1.0 directional_threshold=0 directional_scale=0.5 score_probe=64/task score_source=replay_buffer"
PORT=$((30500 + RANDOM % 1000))
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/lifelong_adaptation.py \
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=BriefPolicy \
  lifelong=mlr_zero_threshold_directional_guard folder="$DATA_ROOT" pretrain="$CHECKPOINT" \
  train.n_epochs=2 train.batch_size=10 train.num_workers=0 \
  train.optimizer.kwargs.lr=1e-4 \
  +start_task=6 +end_task=8 +aug=false +feature_dir="$FEATURE_ROOT" \
  +buffer_dir="$PROJECT_ROOT/buffers/${SMOKE_NAME}" \
  +experiment_root="$PROJECT_ROOT/runs/${SMOKE_NAME}" \
  +past_data_rate=0.5 +ifa_enabled=false +ifa_weight=0.1 +gamma=0.31 \
  +robot_igfa_gate_strength=1.0 +temporal_bin_map="$TEMPORAL_BIN_MAP" \
  +directional_guard_threshold=0.0 +directional_guard_scale=0.5 \
  +directional_guard_score_probe_size=64 \
  +directional_guard_score_probe_batch_size=32 \
  +save_final_checkpoint=true training=true
