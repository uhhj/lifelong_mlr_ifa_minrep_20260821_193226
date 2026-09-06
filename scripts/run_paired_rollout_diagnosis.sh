#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" LIBERO_CONFIG_PATH="$PWD/data/libero_config"
export HF_HOME="$PWD/data/huggingface_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python}"
RUN="$PWD/runs/paired_rollout_diagnosis"
mkdir -p "$RUN" reports/paired_rollout_diagnosis
exec 9>"$RUN/queue.lock"
flock -n 9
run_case() {
  local seed="$1" stage="$2" task="$3" arm="$4" mode="$5" output="$6" model
  if [[ -f "$output/COMPLETE.json" ]]; then return; fi
  if [[ "$arm" == corrected_ifa ]]; then
    model="$PWD/runs/own_reference_control/ifa/seed${seed}/run_001"
  elif [[ "$arm" == release_ifa ]]; then
    model="$PWD/runs/final_paper_text_faithful/ifa/seed${seed}/run_001"
  else
    model="$PWD/runs/final_paper_text_faithful/mlr/seed${seed}/run_001"
  fi
  mkdir -p "$output"
  printf 'RUNNING seed=%s stage=%s task=%s arm=%s mode=%s %s\n' "$seed" "$stage" "$task" "$arm" "$mode" "$(date -Is)" > "$RUN/status.txt"
  ROLLOUT_TRACE_MODE="$mode" ROLLOUT_TRACE_DIR="$output" "$PYTHON" scripts/trace_paired_rollout.py \
    seed="$seed" benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
    folder="$PWD/data" bddl_folder="$PWD/libero/libero/bddl_files" \
    init_states_folder="$PWD/libero/libero/init_files" pretrain="$PWD/checkpoints/object_multitask_model_ep99.pth" \
    eval.n_eval=20 eval.num_procs=20 eval.max_steps=600 +start_task=6 +end_task=10 +aug=true \
    +epoch_dir="$model" +test_log_file="$output/success.log" training=false \
    +eval_stage="$stage" +eval_task_ids="[$task]" '+eval_epochs=[99]' \
    +experiment_root="$output/config" > "$output/stdout.log" 2>&1
}
run_case 43 7 7 mlr light "$RUN/parity_light"
run_case 43 7 7 mlr full "$RUN/seed43_stage7_task7/mlr"
"$PYTHON" scripts/check_paired_rollout_parity.py
for spec in '43 7 7' '42 9 7' '43 9 9'; do
  read -r seed stage task <<< "$spec"
  for arm in mlr release_ifa corrected_ifa; do
    run_case "$seed" "$stage" "$task" "$arm" full "$RUN/seed${seed}_stage${stage}_task${task}/$arm"
  done
done
printf 'ROLLOUTS_COMPLETE %s\n' "$(date -Is)" > "$RUN/status.txt"
