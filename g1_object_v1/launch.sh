#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .migration/activate.sh
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -m g1_object_v1.queue
