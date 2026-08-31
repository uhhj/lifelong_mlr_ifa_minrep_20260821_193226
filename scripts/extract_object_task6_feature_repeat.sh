#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

FEATURE_REPEAT_ROOT="$PROJECT_ROOT/features/paper_v2_object_seed42_repeat64"
test ! -e "$FEATURE_REPEAT_ROOT/6" || {
  echo "Refusing to overwrite repeat cache: $FEATURE_REPEAT_ROOT/6" >&2
  exit 1
}
mkdir -p "$FEATURE_REPEAT_ROOT"

PORT=$((29500 + RANDOM % 1000))
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/save_feature.py \
  seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" \
  train.batch_size=1 train.num_workers=0 pretrain="$CHECKPOINT" \
  +start_task=6 +end_task=7 +aug=true +max_feature_batches=64 \
  +feature_dir="$FEATURE_REPEAT_ROOT" \
  +experiment_root="$PROJECT_ROOT/runs/paper_v2_object_seed42_feature_repeat64" \
  training=false

FEATURE_REPEAT_ROOT="$FEATURE_REPEAT_ROOT" "$PYTHON" scripts/audit_object_feature_cache.py
