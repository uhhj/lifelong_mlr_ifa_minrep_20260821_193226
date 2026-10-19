#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
export CORESET_SOURCE=/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed43/source CORESET_OUT=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed43_20261019 CORESET_SEED=43
export RA_PARENT=/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1 RA_PARENT_SOURCE=/data/mlr_ifa/experiment_workspaces/Seed43_three_arm_repeat_20261016_v1/rc/source
export PYTHONPATH="$CORESET_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"
