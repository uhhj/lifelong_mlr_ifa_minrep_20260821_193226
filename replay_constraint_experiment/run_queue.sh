#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON=/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python
export CUDA_VISIBLE_DEVICES=0
source scripts/_common.sh
PKG="$PROJECT_ROOT/replay_constraint_experiment"
DRIVER="$PROJECT_ROOT/runs/replay_constraint_driver"
mkdir -p "$DRIVER"
exec 9>"$DRIVER/queue.lock"
flock -n 9 || exit 1
phase=START
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "FAILED $phase rc=$rc $(date -Is)" > "$DRIVER/status.txt"; fi' EXIT
for SEED in 42 43 44; do
  phase="TRAIN_SEED${SEED}"
  echo "$phase $(date -Is)" > "$DRIVER/status.txt"
  "$PYTHON" "$PKG/run.py" train --repo "$PROJECT_ROOT" --arm constraint --seed "$SEED" --gpu 0
  "$PYTHON" "$PKG/validate_run.py" "$PROJECT_ROOT/runs/replay_constraint/aggregate_v1/formal/constraint/seed${SEED}"
  phase="EVAL_SEED${SEED}"
  echo "$phase $(date -Is)" > "$DRIVER/status.txt"
  "$PYTHON" "$PKG/run.py" eval --repo "$PROJECT_ROOT" --arm constraint --seed "$SEED" --gpu 0
done
phase=SUMMARY
echo "$phase $(date -Is)" > "$DRIVER/status.txt"
REF_ARGS=()
for SEED in 42 43 44; do
  REF_ARGS+=(--reference "mlr:$SEED:$PROJECT_ROOT/runs/final_paper_text_faithful/mlr/seed$SEED/primary_ep99.log")
  REF_ARGS+=(--reference "ifa_corrected:$SEED:$PROJECT_ROOT/runs/own_reference_control/ifa/seed$SEED/primary_ep99.log")
done
"$PYTHON" "$PKG/summarize.py" --run-root "$PROJECT_ROOT/runs/replay_constraint/aggregate_v1/formal" "${REF_ARGS[@]}" --output "$PROJECT_ROOT/reports/replay_constraint/aggregate_v1_results.json"
echo "COMPLETE $(date -Is)" > "$DRIVER/status.txt"
