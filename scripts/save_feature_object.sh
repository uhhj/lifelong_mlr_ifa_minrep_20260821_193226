#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

PORT=$((29500 + RANDOM % 1000))
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/save_feature.py \
  seed=42 \
  benchmark_name=LIBERO_OBJECT \
  policy=complete_policy \
  lifelong=basic \
  folder="$DATA_ROOT" \
  bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" \
  train.batch_size=1 \
  train.num_workers=0 \
  pretrain="$CHECKPOINT" \
  +start_task=6 \
  +end_task=10 \
  +aug=true \
  +feature_dir="$FEATURE_ROOT" \
  +experiment_root="$PROJECT_ROOT/runs/object_seed42_feature" \
  training=false
