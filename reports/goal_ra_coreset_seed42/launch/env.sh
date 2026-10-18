#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
export CORESET_SOURCE=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/source CORESET_OUT=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018 CORESET_SEED=42
export RA_PARENT=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017 RA_PARENT_SOURCE=/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed42/source
export PYTHONPATH="$CORESET_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"

export SC_BINDING_SOURCE=/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018
