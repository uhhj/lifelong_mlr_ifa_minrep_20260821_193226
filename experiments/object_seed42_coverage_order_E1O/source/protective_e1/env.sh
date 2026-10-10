#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
E1_SOURCE=/data/mlr_ifa/experiment_workspaces/E1O_coverage_order_20261009/source
E1_OUT=/data/mlr_ifa/experiments/E1O_coverage_order_seed42_20261009
export E1_SOURCE E1_OUT
export PYTHONPATH="$E1_SOURCE/bootstrap:$E1_SOURCE:$E1_SOURCE/libero"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
cd "$E1_SOURCE"
