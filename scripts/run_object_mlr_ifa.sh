#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs
test -d "$FEATURE_ROOT/9" || { echo "Missing formal task 6-9 latent features: $FEATURE_ROOT" >&2; exit 1; }

echo "[protocol] phase=train method=mlr_ifa seed=$SEED gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=6-9 epochs=100 batch_size=10 lr=1e-4 lambda_ifa=0.1 gamma=0.31 feature_dir=$FEATURE_ROOT"

PORT=$((29500 + RANDOM % 1000))
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/lifelong_adaptation.py \
  seed="$SEED" \
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
  +buffer_dir="$PROJECT_ROOT/buffers/${RUN_PREFIX}_mlr_ifa" \
  +experiment_root="$PROJECT_ROOT/runs/${RUN_PREFIX}_mlr_ifa" \
  +past_data_rate=0.5 \
  +ifa_enabled=true \
  +ifa_weight=0.1 \
  +gamma=0.31 \
  training=true
