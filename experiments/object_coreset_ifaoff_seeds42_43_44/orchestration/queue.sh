#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/activate.sh
W=/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014
Q=/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
exec 8>"$Q/queue.lock"
flock -n 8
test ! -e "$Q/queue.started"
date -Is > "$Q/queue.started"
echo $$ > "$Q/queue.pid"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; echo "$rc" > "$Q/queue.exit"' EXIT
run() {
 local label="$1"; shift
 printf '%s START %s\n' "$(date -Is)" "$label" >> "$Q/commands.log"
 set +e
 "$@" > "$Q/$label.log" 2>&1
 local rc=$?
 set -e
 printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$rc" >> "$Q/commands.log"
 test "$rc" -eq 0
}
for s in 42 43 44; do
 run "seed${s}_pipeline" bash "$W/original_protocol/seed_pipeline.sh" "$s"
 export CORESET_SOURCE="$W/noifa/seed$s/source" CORESET_OUT="/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed${s}_20261014" CORESET_SEED="$s"
 export PYTHONPATH="$CORESET_SOURCE"
 cd "$CORESET_SOURCE"
 run "seed${s}_acceptance" /data/mlr_ifa/env/bin/python -m coreset_exp.acceptance
done
run compare /data/mlr_ifa/env/bin/python "$W/original_protocol/compare.py"
