#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:?usage: scripts/run_final_paper_text_faithful.sh protocol|test|smoke|train|train-seed|eval-primary|eval-secondary|summarize|formal}"
DRIVER_DIR="$PROJECT_ROOT/runs/final_paper_text_faithful_driver"
mkdir -p "$DRIVER_DIR"

protocol() {
  "$PYTHON" scripts/create_final_paper_text_protocol.py
}

run_tests() {
  "$PYTHON" -m pytest -q tests/test_final_paper_text_faithful.py
  printf 'TESTS_PASS %s\n' "$(date -Is)" > "$DRIVER_DIR/TESTS_PASS"
}

safe_clear_smoke() {
  local smoke_root="/tmp/final_paper_text_faithful_smoke"
  local resolved
  resolved="$(realpath -m "$smoke_root")"
  [[ "$resolved" == "/tmp/final_paper_text_faithful_smoke" ]]
  rm -rf -- "$resolved"
}

smoke_train() {
  local method="$1" end_task="$2" smoke_root="$3"
  local ifa=false
  [[ "$method" == ifa ]] && ifa=true
  mkdir -p "$smoke_root/run" "$smoke_root/buffer"
  local port=$((31500 + RANDOM % 1000))
  echo "[smoke] method=$method seed=42 tasks=6-$((end_task - 1)) max_optimizer_updates=20 batch=10 gamma=0.30"
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$port" \
    libero/lifelong/lifelong_adaptation.py \
    seed=42 benchmark_name=LIBERO_OBJECT policy=BriefPolicy lifelong=mlr \
    folder="$DATA_ROOT" pretrain="$CHECKPOINT" \
    train.n_epochs=100 train.batch_size=10 train.num_workers=0 \
    train.optimizer.kwargs.lr=1e-4 train.optimizer.kwargs.betas='[0.9,0.999]' \
    train.optimizer.kwargs.eps=1e-8 train.optimizer.kwargs.weight_decay=0.1 \
    +start_task=6 +end_task="$end_task" +aug=false \
    +feature_dir="$PROJECT_ROOT/features/object_seed42" +buffer_dir="$smoke_root/buffer" \
    +experiment_root="$smoke_root/run" +past_data_rate=0.5 \
    +ifa_enabled="$ifa" +ifa_weight=0.1 +gamma=0.30 \
    +repro_scheduler_mode=batch_step +repro_arm="final_paper_text_${method}_smoke" \
    +max_optimizer_updates=20 +save_final_checkpoint=true training=true \
    2>&1 | tee "$smoke_root/${method}.log"
}

run_smoke() {
  require_object_inputs
  local smoke_root="/tmp/final_paper_text_faithful_smoke"
  safe_clear_smoke
  mkdir -p "$smoke_root/mlr" "$smoke_root/ifa"
  smoke_train mlr 7 "$smoke_root/mlr"
  smoke_train ifa 8 "$smoke_root/ifa"
  grep -Fq 'stage=7 ifa_enabled=True selected_pairs=[(6, 7)]' "$smoke_root/ifa/ifa.log"
  "$PYTHON" - "$smoke_root" <<'PY'
import csv, math, shutil, sys
from pathlib import Path
from libero.lifelong.algos.mlr import should_save_mlr_checkpoint

root = Path(sys.argv[1])
assert should_save_mlr_checkpoint(99, 100, (), True)
for method, expected_rows in (("mlr", 1), ("ifa", 2)):
    run = root / method / "run" / "run_001"
    rows = list(csv.DictReader((run / "paper_repro_train_curve.csv").open()))
    assert len(rows) == expected_rows
    assert all(math.isfinite(float(row["mean_bc_loss"])) for row in rows)
    assert all(row["scheduler_mode"] == "batch_step" for row in rows)
    assert all(int(row["batch_size"]) == 10 for row in rows)
    lr_rows = list(csv.DictReader((run / "lr_protocol.csv").open()))
    assert all(int(row["optimizer_step_end"]) == 20 for row in lr_rows)
    assert all(int(row["warmup_steps"]) == 500 for row in lr_rows)
    assert all(0 < float(row["lr_end"]) <= 1e-4 for row in lr_rows)
source = root / "mlr" / "run" / "run_001" / "mlrifa_6_model_ep0.pth"
target = source.with_name("mlrifa_6_model_ep99.pth")
assert source.is_file()
shutil.copy2(source, target)
PY
  local eval_log="$smoke_root/primary_ep99_eval.log"
  "$PYTHON" libero/lifelong/test.py \
    seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
    folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
    init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT" \
    eval.n_eval=1 eval.num_procs=1 +start_task=6 +end_task=7 +aug=true \
    +eval_stage=6 '+eval_task_ids=[6]' '+eval_epochs=[99]' \
    +epoch_dir="$smoke_root/mlr/run/run_001" +test_log_file="$eval_log" \
    +experiment_root="$smoke_root/eval" training=false
  grep -Eq '^\[task 6 epoch 99 and 6 succ\.\]' "$eval_log"
  safe_clear_smoke
  printf 'SMOKE_PASS %s mlr_task6_updates=20 ifa_through_task7_updates_per_stage=20 primary_eval_trials=1\n' "$(date -Is)" > "$DRIVER_DIR/SMOKE_PASS"
}

run_two() {
  local label1="$1" gpu1="$2" label2="$3" gpu2="$4"
  shift 4
  local command=("$@")
  CUDA_VISIBLE_DEVICES="$gpu1" MUJOCO_EGL_DEVICE_ID="$gpu1" "${command[@]}" > "$DRIVER_DIR/${label1}.log" 2>&1 &
  local pid1=$!
  CUDA_VISIBLE_DEVICES="$gpu2" MUJOCO_EGL_DEVICE_ID="$gpu2" "${command[@]}" > "$DRIVER_DIR/${label2}.log" 2>&1 &
  local pid2=$!
  local rc=0
  wait "$pid1" || rc=1
  wait "$pid2" || rc=1
  return "$rc"
}

train_job() {
  local method="$1" seed="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" METHOD="$method" SEED="$seed" \
    bash scripts/run_object_final_paper_text_train.sh "$method" "$seed"
}

eval_job() {
  local method="$1" seed="$2" view="$3" gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" METHOD="$method" SEED="$seed" VIEW="$view" \
    bash scripts/eval_object_final_paper_text.sh "$method" "$seed" "$view"
}

run_all_training() {
  echo "MLR_SEEDS42_43 $(date -Is)" > "$DRIVER_DIR/status.txt"
  train_job mlr 42 0 > "$DRIVER_DIR/train_mlr_seed42.log" 2>&1 & local p1=$!
  train_job mlr 43 1 > "$DRIVER_DIR/train_mlr_seed43.log" 2>&1 & local p2=$!
  local rc=0; wait "$p1" || rc=1; wait "$p2" || rc=1; [[ "$rc" -eq 0 ]]
  echo "MLR_SEED44 $(date -Is)" > "$DRIVER_DIR/status.txt"
  train_job mlr 44 0 > "$DRIVER_DIR/train_mlr_seed44.log" 2>&1
  echo "IFA_SEEDS42_43 $(date -Is)" > "$DRIVER_DIR/status.txt"
  train_job ifa 42 0 > "$DRIVER_DIR/train_ifa_seed42.log" 2>&1 & p1=$!
  train_job ifa 43 1 > "$DRIVER_DIR/train_ifa_seed43.log" 2>&1 & p2=$!
  rc=0; wait "$p1" || rc=1; wait "$p2" || rc=1; [[ "$rc" -eq 0 ]]
  echo "IFA_SEED44 $(date -Is)" > "$DRIVER_DIR/status.txt"
  train_job ifa 44 0 > "$DRIVER_DIR/train_ifa_seed44.log" 2>&1
}

run_all_eval() {
  local view="$1"
  local jobs=(mlr:42 mlr:43 mlr:44 ifa:42 ifa:43 ifa:44)
  local index=0
  while [[ "$index" -lt "${#jobs[@]}" ]]; do
    IFS=: read -r method1 seed1 <<< "${jobs[$index]}"
    IFS=: read -r method2 seed2 <<< "${jobs[$((index + 1))]}"
    echo "EVAL_${view^^}_${method1^^}_SEED${seed1}_${method2^^}_SEED${seed2} $(date -Is)" > "$DRIVER_DIR/status.txt"
    eval_job "$method1" "$seed1" "$view" 0 > "$DRIVER_DIR/eval_${view}_${method1}_seed${seed1}.log" 2>&1 & local p1=$!
    eval_job "$method2" "$seed2" "$view" 1 > "$DRIVER_DIR/eval_${view}_${method2}_seed${seed2}.log" 2>&1 & local p2=$!
    local rc=0; wait "$p1" || rc=1; wait "$p2" || rc=1; [[ "$rc" -eq 0 ]]
    index=$((index + 2))
  done
}

require_formal_gate() {
  test -s "$DRIVER_DIR/TESTS_PASS"
  test -s "$DRIVER_DIR/SMOKE_PASS"
  test -s "$PROJECT_ROOT/reports/final_paper_text_faithful/PROTOCOL_FROZEN.txt"
  git diff --quiet -- reports/final_paper_text_faithful/PROTOCOL.json reports/final_paper_text_faithful/PROTOCOL.md reports/final_paper_text_faithful/PROTOCOL_FROZEN.txt
}

run_formal() {
  require_formal_gate
  local lock="$DRIVER_DIR/FORMAL_LOCK"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "Formal driver already running: $lock" >&2
    exit 1
  fi
  trap 'rmdir "$lock" 2>/dev/null || true' EXIT
  if [[ -s "$DRIVER_DIR/FORMAL_COMPLETE" ]]; then
    echo "Formal execution already complete"
    return
  fi
  run_all_training
  run_all_eval primary
  run_all_eval secondary
  echo "SUMMARIZE $(date -Is)" > "$DRIVER_DIR/status.txt"
  "$PYTHON" scripts/summarize_final_paper_text_faithful.py > "$DRIVER_DIR/summarize.log" 2>&1
  printf 'FORMAL_COMPLETE %s\n' "$(date -Is)" > "$DRIVER_DIR/FORMAL_COMPLETE"
  echo "COMPLETE $(date -Is)" > "$DRIVER_DIR/status.txt"
}

case "$MODE" in
  protocol) protocol ;;
  test) run_tests ;;
  smoke) run_smoke ;;
  train) require_formal_gate; run_all_training ;;
  train-seed) require_formal_gate; train_job "${METHOD:?METHOD required}" "${SEED:?SEED required}" "${CUDA_VISIBLE_DEVICES:-0}" ;;
  eval-primary) require_formal_gate; run_all_eval primary ;;
  eval-secondary) require_formal_gate; run_all_eval secondary ;;
  summarize) "$PYTHON" scripts/summarize_final_paper_text_faithful.py ;;
  formal) run_formal ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
