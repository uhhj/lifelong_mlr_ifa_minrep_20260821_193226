#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate /home/dm/miniconda3/envs/wzp_mlr_tmp

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${FORMAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$PROJECT_ROOT/runs/zero_guard_formal_overnight_${STAMP}"
STATUS="$LOG_ROOT/status.tsv"
mkdir -p "$LOG_ROOT"
printf 'timestamp\tseed\tphase\tstatus\tdetail\n' > "$STATUS"

record() {
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" >> "$STATUS"
}

fail_pipeline() {
  record pipeline orchestration FAILED "line=$1 exit=$2"
}
trap 'fail_pipeline "$LINENO" "$?"' ERR

test "$(git branch --show-current)" = \
  "research/zero-threshold-directional-guard-formal"
test -z "$(git status --porcelain)"
SOURCE_SHA="$(git rev-parse HEAD)"
printf '%s\n' "$SOURCE_SHA" > "$LOG_ROOT/formal_experiment_source_sha.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader > "$LOG_ROOT/gpu_preflight.csv"
df -h "$PROJECT_ROOT" > "$LOG_ROOT/disk_preflight.txt"

for seed in 42 43 44; do
  run="$PROJECT_ROOT/runs/object_seed${seed}_mlr_zero_directional_guard"
  buffer="$PROJECT_ROOT/buffers/object_seed${seed}_mlr_zero_directional_guard"
  if [[ -e "$run" || -e "$buffer" ]]; then
    record "$seed" preflight FAILED "formal run or buffer already exists"
    exit 1
  fi
done

run_one() {
  local seed="$1"
  local gpu="$2"
  local train_log="$LOG_ROOT/seed${seed}_train.log"
  local eval_log="$LOG_ROOT/seed${seed}_eval_driver.log"
  record "$seed" train STARTED "gpu=$gpu source_sha=$SOURCE_SHA"
  if ! CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" SEED="$seed" \
    bash scripts/run_object_mlr_zero_directional_guard.sh \
    > "$train_log" 2>&1; then
    record "$seed" train FAILED "gpu=$gpu log=$train_log"
    return 1
  fi
  record "$seed" train COMPLETED "gpu=$gpu"
  record "$seed" eval STARTED "gpu=$gpu trials=20"
  if ! CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" SEED="$seed" \
    bash scripts/eval_object.sh mlr_zero_directional_guard \
    > "$eval_log" 2>&1; then
    record "$seed" eval FAILED "gpu=$gpu log=$eval_log"
    return 1
  fi
  record "$seed" eval COMPLETED "gpu=$gpu records=30"
}

record pipeline formal STARTED "source_sha=$SOURCE_SHA"
(
  set -e
  run_one 42 0
) &
LANE0=$!
(
  set -e
  run_one 43 1
  run_one 44 1
) &
LANE1=$!

lane_failed=0
wait "$LANE0" || lane_failed=1
wait "$LANE1" || lane_failed=1
if [[ "$lane_failed" -ne 0 ]]; then
  record pipeline formal FAILED "one or more GPU lanes failed"
  exit 1
fi

record pipeline metrics STARTED "seeds=42,43,44"
python scripts/compute_zero_threshold_directional_guard_formal_metrics.py \
  --runs-root runs \
  --temporal-v1-json \
    reports/temporal_robot_igfa_v1/temporal_robot_igfa_3seed_metrics.json \
  --baseline-json \
    reports/baseline_reconciliation/object_3seed_metrics.json \
  --output-dir reports/zero_threshold_directional_guard_formal \
  --seeds 42 43 44 \
  > "$LOG_ROOT/metrics.log" 2>&1
record pipeline metrics COMPLETED "reports generated"

git add reports/zero_threshold_directional_guard_formal
git commit -m "Record formal directional guard results"
if git push origin research/zero-threshold-directional-guard-formal \
  > "$LOG_ROOT/git_push.log" 2>&1; then
  record pipeline git COMPLETED "results committed and pushed"
else
  record pipeline git DEFERRED \
    "results committed locally; server GitHub transport failed"
fi
record pipeline formal COMPLETED "final_sha=$(git rev-parse HEAD)"
