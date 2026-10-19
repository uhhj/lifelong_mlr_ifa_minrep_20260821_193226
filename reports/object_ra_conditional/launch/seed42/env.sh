#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
export CORESET_SOURCE=/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed42/source CORESET_OUT=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019 CORESET_SEED=42
export RA_PARENT=/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013 RA_PARENT_SOURCE=/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed42_20261013/source
export PYTHONPATH="$CORESET_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"
