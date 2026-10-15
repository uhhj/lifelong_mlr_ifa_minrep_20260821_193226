#!/usr/bin/env bash
set -euo pipefail
s="$1"
source /data/mlr_ifa/activate.sh
W=/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014
Q=/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014
export CORESET_SOURCE="$W/noifa/seed$s/source" CORESET_OUT="/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed${s}_20261014" CORESET_SEED="$s"
export PYTHONPATH="$CORESET_SOURCE" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"
exec 9>"$CORESET_OUT/pipeline.lock"
flock -n 9
test ! -e "$CORESET_OUT/pipeline.started"
date -Is > "$CORESET_OUT/pipeline.started"
echo $$ > "$CORESET_OUT/pipeline.pid"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; echo "$rc" > "$CORESET_OUT/pipeline.exit"' EXIT
run() {
 local label="$1"; shift
 printf '%s START %s ' "$(date -Is)" "$label" >> "$CORESET_OUT/commands.log"
 printf '%q ' "$@" >> "$CORESET_OUT/commands.log"
 printf '\n' >> "$CORESET_OUT/commands.log"
 set +e
 "$@" > "$CORESET_OUT/$label.log" 2>&1
 local rc=$?
 set -e
 printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$rc" >> "$CORESET_OUT/commands.log"
 test "$rc" -eq 0
}
run contract /data/mlr_ifa/env/bin/python "$W/original_protocol/guard.py"
run software_tests /data/mlr_ifa/env/bin/python -m unittest discover -s coreset_exp -t . -p 'test_*.py' -v
run spawn_check /data/mlr_ifa/env/bin/python "$W/original_protocol/spawn_check.py"
run gpu_preflight /data/mlr_ifa/env/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 "$W/preflight_noifa.py"
for a in random coreset; do
 run "train_$a" /data/mlr_ifa/env/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 -m coreset_exp.train "$a"
done
run training_accept /data/mlr_ifa/env/bin/python "$W/accept_noifa_training.py" "$s"
run eval_contract /data/mlr_ifa/env/bin/python "$W/original_protocol/guard.py"
for a in random coreset; do
 run "evaluate_$a" /data/mlr_ifa/env/bin/python -u -m coreset_exp.evaluate "$a"
done
run summarize /data/mlr_ifa/env/bin/python -m coreset_exp.summarize
