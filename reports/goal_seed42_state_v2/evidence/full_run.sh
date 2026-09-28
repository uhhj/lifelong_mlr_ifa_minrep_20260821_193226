#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/activate.sh
base=/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2
cd "$base/source"
bash scripts/run_goal_seed42.sh --mode full --pretrained /data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth --data-root /data/mlr_ifa/datasets --bddl-root /data/mlr_ifa/src/libero/libero/bddl_files --init-states-root /data/mlr_ifa/src/libero/libero/init_files --run-dir "$base/run" --smoke-dir "$base/smoke_v2_release3"
