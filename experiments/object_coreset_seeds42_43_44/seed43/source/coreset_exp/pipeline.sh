#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed43_20261013/source/coreset_exp/env.sh
exec 9>"$CORESET_OUT/pipeline.lock"
flock -n 9 || exit 70
test ! -e "$CORESET_OUT/pipeline.started"
date -Is > "$CORESET_OUT/pipeline.started"
echo $$ > "$CORESET_OUT/pipeline.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$CORESET_OUT/pipeline.exit"' EXIT
run() {
  label="$1"; shift
  printf '%s START %s ' "$(date -Is)" "$label" >> "$CORESET_OUT/commands.log"
  printf '%q ' "$@" >> "$CORESET_OUT/commands.log"
  printf '\n' >> "$CORESET_OUT/commands.log"
  set +e
  "$@" > "$CORESET_OUT/$label.log" 2>&1
  rc=$?
  set -e
  printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$rc" >> "$CORESET_OUT/commands.log"
  test "$rc" = 0 || exit "$rc"
}
run export torchrun --nproc_per_node=1 --master_port=38316 -m coreset_exp.export
run train_random torchrun --nproc_per_node=1 --master_port=38316 -m coreset_exp.train random
run train_coreset torchrun --nproc_per_node=1 --master_port=38316 -m coreset_exp.train coreset
run evaluate_random python -u -m coreset_exp.evaluate random
run evaluate_coreset python -u -m coreset_exp.evaluate coreset
run summarize python -u -m coreset_exp.summarize
