#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec 9>"$E1_OUT/process.lock"
flock -n 9
test ! -e "$E1_OUT/features/6/0.pt"
mkdir -p "$E1_OUT/features" "$E1_OUT/export"
printf '%s\n' "$$" > "$E1_OUT/export.pid"
trap 'code=$?; printf "%s\n" "$code" > "$E1_OUT/export.exit"' EXIT
python -m torch.distributed.run --nproc_per_node=1 --master_port=37983 -m protective_e1.export \
 seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
 folder=/data/mlr_ifa/datasets bddl_folder="$E1_SOURCE/libero/libero/bddl_files" \
 init_states_folder="$E1_SOURCE/libero/libero/init_files" \
 train.batch_size=1 train.num_workers=0 pretrain=/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth \
 +start_task=6 +end_task=10 +aug=true +feature_dir="$E1_OUT/features" \
 +experiment_root="$E1_OUT/export" training=false use_wandb=false
