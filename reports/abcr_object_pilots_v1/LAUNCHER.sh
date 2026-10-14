#!/bin/bash
set -euo pipefail
cd /data/lifelong_mlr_ifa_minrep_20260912
export PYTHONPATH=/tmp/abcr_object_pilots_runtime:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>reports/abcr_object_pilots_v1/RUN.lock
flock -n 9 || exit 20
python -u -m abcr_object_pilots_v1.encode --repo "$PWD" --out reports/abcr_object_pilots_v1 --mode language
python -u -m abcr_object_pilots_v1.encode --repo "$PWD" --out reports/abcr_object_pilots_v1 --mode vision
python -u -m abcr_object_pilots_v1.analyze --out reports/abcr_object_pilots_v1
python -u -m abcr_object_pilots_v1.report --out reports/abcr_object_pilots_v1
