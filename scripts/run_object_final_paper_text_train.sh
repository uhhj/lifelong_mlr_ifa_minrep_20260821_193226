#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_object_inputs

METHOD="${METHOD:-${1:-}}"
SEED="${SEED:-${2:-}}"
test -n "$METHOD" && test -n "$SEED" || { echo "METHOD and SEED are required" >&2; exit 2; }
case "$METHOD" in
  mlr) IFA_ENABLED=false ;;
  ifa) IFA_ENABLED=true ;;
  *) echo "METHOD must be mlr or ifa" >&2; exit 2 ;;
esac
case "$SEED" in 42|43|44) ;; *) echo "SEED must be 42, 43, or 44" >&2; exit 2 ;; esac

PROTOCOL_ROOT="$PROJECT_ROOT/reports/final_paper_text_faithful"
test -s "$PROTOCOL_ROOT/PROTOCOL_FROZEN.txt"
git diff --quiet -- "$PROTOCOL_ROOT/PROTOCOL.json" "$PROTOCOL_ROOT/PROTOCOL.md" "$PROTOCOL_ROOT/PROTOCOL_FROZEN.txt"

FEATURE_ROOT="$PROJECT_ROOT/features/object_seed${SEED}"
RUN_DIR="$PROJECT_ROOT/runs/final_paper_text_faithful/$METHOD/seed${SEED}"
BUFFER_DIR="$PROJECT_ROOT/buffers/final_paper_text_faithful/$METHOD/seed${SEED}"
SENTINEL="$RUN_DIR/TRAINING_COMPLETE"
ACTIVE="$RUN_DIR/TRAINING_ACTIVE"
test -d "$FEATURE_ROOT/9" || { echo "Missing feature cache: $FEATURE_ROOT" >&2; exit 1; }
if [[ -s "$SENTINEL" ]]; then
  echo "$METHOD seed$SEED training already complete"
  exit 0
fi
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -maxdepth 1 -type d -name 'run_*' -print -quit | grep -q .; then
  if [[ -s "$ACTIVE" ]] && ! kill -0 "$(cat "$ACTIVE")" 2>/dev/null; then
    resolved_run="$(realpath -m "$RUN_DIR")"
    resolved_buffer="$(realpath -m "$BUFFER_DIR")"
    [[ "$resolved_run" == "$PROJECT_ROOT/runs/final_paper_text_faithful/$METHOD/seed$SEED" ]]
    [[ "$resolved_buffer" == "$PROJECT_ROOT/buffers/final_paper_text_faithful/$METHOD/seed$SEED" ]]
    rm -rf -- "$resolved_run" "$resolved_buffer"
  else
    echo "Incomplete run is unmarked or still active: $RUN_DIR" >&2
    exit 1
  fi
fi
mkdir -p "$RUN_DIR" "$BUFFER_DIR"
printf '%s\n' "$$" > "$ACTIVE"

echo "[protocol] phase=train scope=final_paper_text_faithful method=$METHOD seed=$SEED gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=6-9 epochs=100 batch_size=10 optimizer=AdamW lr=1e-4 scheduler=batch_step warmup=500 ifa_enabled=$IFA_ENABLED lambda=0.1 gamma=0.30 feature=$FEATURE_ROOT checkpoint=$CHECKPOINT"
PORT=$((29500 + RANDOM % 1000))
"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  libero/lifelong/lifelong_adaptation.py \
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=BriefPolicy lifelong=mlr \
  folder="$DATA_ROOT" pretrain="$CHECKPOINT" \
  train.n_epochs=100 train.batch_size=10 train.num_workers=0 \
  train.optimizer.kwargs.lr=1e-4 train.optimizer.kwargs.betas='[0.9,0.999]' \
  train.optimizer.kwargs.eps=1e-8 train.optimizer.kwargs.weight_decay=0.1 \
  +start_task=6 +end_task=10 +aug=false \
  +feature_dir="$FEATURE_ROOT" +buffer_dir="$BUFFER_DIR" \
  +experiment_root="$RUN_DIR" +past_data_rate=0.5 \
  +ifa_enabled="$IFA_ENABLED" +ifa_weight=0.1 +gamma=0.30 \
  +repro_scheduler_mode=batch_step +repro_arm="final_paper_text_${METHOD}" \
  +save_final_checkpoint=true training=true

EPOCH_DIR="$RUN_DIR/run_001"
count="$(find "$EPOCH_DIR" -maxdepth 1 -name 'mlrifa_*_model_ep*.pth' | wc -l)"
test "$count" -eq 28 || { echo "Expected 28 checkpoints, found $count" >&2; exit 1; }
printf 'TRAINING_COMPLETE %s method=%s seed=%s checkpoints=%s\n' "$(date -Is)" "$METHOD" "$SEED" "$count" > "$SENTINEL"
rm -f -- "$ACTIVE"
