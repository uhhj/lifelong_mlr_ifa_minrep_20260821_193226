#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs
test -d "$FEATURE_ROOT/9" || { echo "Missing formal task 6-9 latent features: $FEATURE_ROOT" >&2; exit 1; }

PORT=$((29500 + RANDOM % 1000))
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/lifelong_adaptation.py \
  seed=42 \
  benchmark_name=LIBERO_OBJECT \
  policy=BriefPolicy \
  lifelong=mlr \
  folder="$DATA_ROOT" \
  pretrain="$CHECKPOINT" \
  train.n_epochs=100 \
  train.batch_size=10 \
  train.num_workers=0 \
  train.optimizer.kwargs.lr=1e-4 \
  +start_task=6 \
  +end_task=10 \
  +aug=false \
  +feature_dir="$FEATURE_ROOT" \
  +buffer_dir="$PROJECT_ROOT/buffers/object_seed42_mlr_ifa" \
  +experiment_root="$PROJECT_ROOT/runs/object_seed42_mlr_ifa" \
  +past_data_rate=0.5 \
  +ifa_enabled=true \
  +ifa_weight=0.1 \
  +gamma=0.31 \
  training=true
