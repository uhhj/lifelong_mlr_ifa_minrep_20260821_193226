#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
export CORESET_SOURCE=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019/baseline/source
export CORESET_OUT=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019
export CORESET_SEED=41
export PYTHONPATH="$CORESET_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"
