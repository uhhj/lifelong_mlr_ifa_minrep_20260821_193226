#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

ARM="${1:?usage: scripts/run_object_task6_reconciliation.sh current_legacy|release18_legacy|batch10_linear_step|batch10_constant}"
case "$ARM" in
  current_legacy) BATCH=10; SCHED=legacy_epoch ;;
  release18_legacy) BATCH=18; SCHED=legacy_epoch ;;
  batch10_linear_step) BATCH=10; SCHED=batch_step ;;
  batch10_constant) BATCH=10; SCHED=constant ;;
  *) echo "Unknown task6 arm: $ARM" >&2; exit 2 ;;
esac

test -d "$FEATURE_ROOT/6" || { echo "Missing task6 features: $FEATURE_ROOT/6" >&2; exit 1; }

if [[ "${SMOKE:-0}" == 1 ]]; then
  RUN_TAG="paper_v2_smoke_${SCHED}"
  EPOCHS=2
  EXTRA_ARGS=(+max_optimizer_updates=20 +save_final_checkpoint=true)
else
  RUN_TAG="paper_v2_object_seed${SEED}_task6_${ARM}"
  EPOCHS=100
  EXTRA_ARGS=('+diagnostic_checkpoint_epochs=[9,19,29,39,49,59,69,79,89,99]')
fi

RUN_DIR="$PROJECT_ROOT/runs/$RUN_TAG"
BUFFER_DIR="$PROJECT_ROOT/buffers/$RUN_TAG"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"
if find "$BUFFER_DIR" -maxdepth 1 -name '*.pt' -print -quit | grep -q .; then
  echo "Refusing non-empty isolated buffer: $BUFFER_DIR" >&2
  exit 1
fi

echo "[protocol] phase=train seed=$SEED arm=$ARM gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=6 stage=6 epochs=$EPOCHS batch_size=$BATCH lr=1e-4 scheduler_mode=$SCHED checkpoint=$CHECKPOINT"

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
  train.n_epochs="$EPOCHS" \
  train.batch_size="$BATCH" \
  train.num_workers=0 \
  train.optimizer.kwargs.lr=1e-4 \
  +start_task=6 \
  +end_task=7 \
  +aug=false \
  +feature_dir="$FEATURE_ROOT" \
  +buffer_dir="$BUFFER_DIR" \
  +experiment_root="$RUN_DIR" \
  +past_data_rate=0.5 \
  +ifa_enabled=false \
  +ifa_weight=0.1 \
  +gamma=0.31 \
  +repro_scheduler_mode="$SCHED" \
  +repro_arm="$ARM" \
  "${EXTRA_ARGS[@]}" \
  training=true
