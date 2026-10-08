#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .migration/activate.sh
PYTHON="$(command -v python)"
CHECKPOINT="$PROJECT_ROOT/checkpoints/object_multitask_model_ep99.pth"
DATA_ROOT="$PROJECT_ROOT/data"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless WANDB_MODE=disabled HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false

METHOD=e1
SEED="${1:?seed}"
VIEW="${2:?view}"
case "$METHOD" in e1) ;; *) echo "METHOD must be mlr or ifa" >&2; exit 2 ;; esac
case "$SEED" in 42|43|44) ;; *) echo "SEED must be 42, 43, or 44" >&2; exit 2 ;; esac
case "$VIEW" in primary|secondary) ;; *) echo "VIEW must be primary or secondary" >&2; exit 2 ;; esac

RUN_DIR="$PROJECT_ROOT/runs/e1_object_v1/seed${SEED}"
EPOCH_DIR="$RUN_DIR/run_001"
test -s "$RUN_DIR/TRAINING_COMPLETE.json" || { echo "Training incomplete: $RUN_DIR" >&2; exit 1; }
if [[ "$VIEW" == primary ]]; then
  LOG_FILE="$RUN_DIR/primary_ep99.log"
  SENTINEL="$RUN_DIR/PRIMARY_EVAL_COMPLETE"
else
  LOG_FILE="$RUN_DIR/secondary_selected.log"
  SENTINEL="$RUN_DIR/SECONDARY_EVAL_COMPLETE"
fi
if [[ -s "$SENTINEL" ]]; then
  echo "$METHOD seed$SEED $VIEW evaluation already complete"
  exit 0
fi
if [[ -s "$LOG_FILE" ]]; then cp "$LOG_FILE" "${LOG_FILE}.interrupted.$(date +%s)"; fi
: > "$LOG_FILE"
export E1_TRIAL_LOG="$PROJECT_ROOT/reports/e1_object_v1/seed${SEED}/trials_${VIEW}.jsonl"
if [[ -s "$E1_TRIAL_LOG" ]]; then mv "$E1_TRIAL_LOG" "${E1_TRIAL_LOG}.interrupted.$(date +%s)"; fi

COMMON_ARGS=(
  seed="$SEED" benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files"
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT"
  eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true
  +epoch_dir="$EPOCH_DIR" +test_log_file="$LOG_FILE" training=false
)

if [[ "$VIEW" == primary ]]; then
  echo "[protocol] phase=eval view=PRIMARY_EP99 method=$METHOD seed=$SEED trials=20 epoch=99 action=release_gmm_sampling"
  "$PYTHON" -m e1_object_v1.evaluate_spawn_v1 "${COMMON_ARGS[@]}" \
    '+eval_epochs=[99]' \
    +experiment_root="$RUN_DIR/eval_primary_ep99"
  "$PYTHON" - "$LOG_FILE" <<'PY'
import re, sys
p = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
records = [tuple(m.groups()) for line in open(sys.argv[1], encoding="utf-8") if (m := p.search(line))]
expected = {(str(task), "99", str(stage)) for stage in range(6, 10) for task in range(6, stage + 1)}
assert len(records) == 10 and {(t, e, s) for t, e, s, _ in records} == expected
PY
else
  echo "[protocol] phase=eval view=SECONDARY_RELEASE_COMPATIBLE_SELECTION method=$METHOD seed=$SEED trials=20 candidates=70,75,80,85,90,95,99"
  "$PYTHON" -m e1_object_v1.evaluate_spawn_v1 "${COMMON_ARGS[@]}" \
    +eval_current_only=true '+eval_epochs=[70,75,80,85,90,95,99]' \
    +experiment_root="$RUN_DIR/eval_secondary_selection"
  read -r E6 E7 E8 E9 < <("$PYTHON" - "$LOG_FILE" <<'PY'
import re, sys
p = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
r = {}
for line in open(sys.argv[1], encoding="utf-8"):
    if m := p.search(line):
        task, epoch, stage = map(int, m.group(1, 2, 3)); r[(stage, epoch, task)] = float(m.group(4))
epochs = (70, 75, 80, 85, 90, 95, 99)
print(*(max(epochs, key=lambda e: (r[(stage, e, stage)], e)) for stage in range(6, 10)))
PY
  )
  for SPEC in "7:$E7:[6]" "8:$E8:[6,7]" "9:$E9:[6,7,8]"; do
    IFS=: read -r STAGE EPOCH TASKS <<< "$SPEC"
    "$PYTHON" -m e1_object_v1.evaluate_spawn_v1 "${COMMON_ARGS[@]}" \
      +eval_stage="$STAGE" +eval_task_ids="$TASKS" +eval_epochs="[$EPOCH]" \
      +experiment_root="$RUN_DIR/eval_secondary_stage${STAGE}"
  done
  "$PYTHON" - "$LOG_FILE" <<'PY'
import re, sys
p = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
records = [m.groups() for line in open(sys.argv[1], encoding="utf-8") if (m := p.search(line))]
assert len(records) == 34, len(records)
PY
fi
printf '%s_EVAL_COMPLETE %s method=%s seed=%s trials=20\n' "${VIEW^^}" "$(date -Is)" "$METHOD" "$SEED" > "$SENTINEL"
