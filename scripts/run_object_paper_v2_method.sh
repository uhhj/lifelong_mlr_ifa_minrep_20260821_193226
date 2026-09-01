#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

METHOD="${1:?usage: scripts/run_object_paper_v2_method.sh mlr|mlr_ifa SEED}"
SEED="${2:?usage: scripts/run_object_paper_v2_method.sh mlr|mlr_ifa SEED}"
case "$METHOD" in
  mlr) IFA_ENABLED=false; GAMMA=0.31 ;;
  mlr_ifa) IFA_ENABLED=true; GAMMA=0.30 ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

FEATURE_ROOT="$PROJECT_ROOT/features/object_seed${SEED}"
RUN_NAME="object_seed${SEED}_${METHOD}_paper_v2"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
BUFFER_DIR="$PROJECT_ROOT/buffers/$RUN_NAME"
test -d "$FEATURE_ROOT/9" || { echo "Missing seed-specific task 6-9 features: $FEATURE_ROOT" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$BUFFER_DIR"
if find "$BUFFER_DIR" -maxdepth 1 -name '*.pt' -print -quit | grep -q .; then
  echo "Refusing non-empty formal buffer: $BUFFER_DIR" >&2; exit 1
fi
if find "$RUN_DIR" -maxdepth 1 -type d -name 'run_*' -print -quit | grep -q .; then
  echo "Refusing existing formal run: $RUN_DIR" >&2; exit 1
fi

echo "[protocol] phase=train method=$METHOD seed=$SEED gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=6-9 epochs=100 batch_size=10 lr=1e-4 scheduler_mode=batch_step warmup_steps=500 ifa_enabled=$IFA_ENABLED gamma=$GAMMA feature_dir=$FEATURE_ROOT"
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
  +ifa_enabled="$IFA_ENABLED" +ifa_weight=0.1 +gamma="$GAMMA" \
  +repro_scheduler_mode=batch_step +repro_arm="${METHOD}_paper_v2" \
  training=true
