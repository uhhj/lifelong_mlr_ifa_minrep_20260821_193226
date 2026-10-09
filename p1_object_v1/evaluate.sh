#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .migration/activate.sh
PYTHON="$(command -v python)"
CHECKPOINT="$PROJECT_ROOT/checkpoints/object_multitask_model_ep99.pth"
DATA_ROOT="$PROJECT_ROOT/data"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless WANDB_MODE=disabled HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false

METHOD=p1
SEED="${1:?seed}"
VIEW="${2:?view}"
case "$METHOD" in p1) ;; *) echo "METHOD must be mlr or ifa" >&2; exit 2 ;; esac
case "$SEED" in 42|43|44) ;; *) echo "SEED must be 42, 43, or 44" >&2; exit 2 ;; esac
case "$VIEW" in primary) ;; *) echo "VIEW must be primary" >&2; exit 2 ;; esac

RUN_DIR="$PROJECT_ROOT/runs/p1_object_v1/seed${SEED}"
EPOCH_DIR="$RUN_DIR/checkpoints"
test -s "$RUN_DIR/TRAINING_COMPLETE.json" || { echo "Training incomplete: $RUN_DIR" >&2; exit 1; }
LOG_FILE="$RUN_DIR/primary_ep99.log"
SENTINEL="$RUN_DIR/PRIMARY_EVAL_COMPLETE"
if [[ -s "$SENTINEL" ]]; then
  echo "$METHOD seed$SEED $VIEW evaluation already complete"
  exit 0
fi
if [[ -s "$LOG_FILE" ]]; then cp "$LOG_FILE" "${LOG_FILE}.interrupted.$(date +%s)"; fi
: > "$LOG_FILE"
export P1_TRIAL_LOG="$PROJECT_ROOT/reports/p1_object_v1/seed${SEED}/trials_${VIEW}.jsonl"
if [[ -s "$P1_TRIAL_LOG" ]]; then mv "$P1_TRIAL_LOG" "${P1_TRIAL_LOG}.interrupted.$(date +%s)"; fi

COMMON_ARGS=(
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files"
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT"
  eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true
  +epoch_dir="$EPOCH_DIR" +test_log_file="$LOG_FILE" training=false
)

  echo "[protocol] phase=eval view=PRIMARY_EP99 method=$METHOD seed=$SEED trials=20 epoch=99 action=release_gmm_sampling"
  "$PYTHON" -m p1_object_v1.evaluate_spawn_v1 "${COMMON_ARGS[@]}" \
    '+eval_epochs=[99]' \
    +experiment_root="$RUN_DIR/eval_primary_ep99"
  "$PYTHON" - "$LOG_FILE" <<'PY'
import re, sys
p = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
records = [tuple(m.groups()) for line in open(sys.argv[1], encoding="utf-8") if (m := p.search(line))]
expected = {(str(task), "99", str(stage)) for stage in range(6, 10) for task in range(6, stage + 1)}
assert len(records) == 10 and {(t, e, s) for t, e, s, _ in records} == expected
PY
printf '%s_EVAL_COMPLETE %s method=%s seed=%s trials=20\n' "${VIEW^^}" "$(date -Is)" "$METHOD" "$SEED" > "$SENTINEL"
