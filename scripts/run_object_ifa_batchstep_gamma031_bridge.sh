#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

SEED="${1:-${SEED:-42}}"
FEATURE_ROOT="$PROJECT_ROOT/features/object_seed${SEED}"
RUN_NAME="object_seed${SEED}_ifa_batchstep_gamma031_bridge"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
BUFFER_DIR="$PROJECT_ROOT/buffers/$RUN_NAME"
LOG_DIR="$PROJECT_ROOT/reports/ifa_optimization_interaction/bridge/raw"
test -d "$FEATURE_ROOT/9" || { echo "Missing seed-specific task 6-9 features: $FEATURE_ROOT" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$BUFFER_DIR" "$LOG_DIR"
if find "$BUFFER_DIR" -maxdepth 1 -name '*.pt' -print -quit | grep -q .; then
  echo "Refusing non-empty formal buffer: $BUFFER_DIR" >&2; exit 1
fi
if find "$RUN_DIR" -maxdepth 1 -type d -name 'run_*' -print -quit | grep -q .; then
  echo "Refusing existing formal run: $RUN_DIR" >&2; exit 1
fi

echo "[protocol] phase=train method=ifa_batchstep_gamma031_bridge seed=$SEED gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=6-9 epochs=100 batch_size=10 lr=1e-4 scheduler_mode=batch_step warmup_steps=500 ifa_enabled=true ifa_weight=0.1 gamma=0.31 feature_dir=$FEATURE_ROOT"
PORT=$((29500 + RANDOM % 1000))
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/lifelong_adaptation.py \
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=BriefPolicy lifelong=mlr \
  folder="$DATA_ROOT" pretrain="$CHECKPOINT" \
  train.n_epochs=100 train.batch_size=10 train.num_workers=0 \
  train.optimizer.kwargs.lr=1e-4 \
  +start_task=6 +end_task=10 +aug=false +feature_dir="$FEATURE_ROOT" \
  +buffer_dir="$BUFFER_DIR" +experiment_root="$RUN_DIR" +past_data_rate=0.5 \
  +ifa_enabled=true +ifa_weight=0.1 +gamma=0.31 \
  +ifa_interaction_logging=true +ifa_interaction_log_dir="$LOG_DIR" \
  +repro_scheduler_mode=batch_step +repro_arm=ifa_batchstep_gamma031_bridge \
  training=true
