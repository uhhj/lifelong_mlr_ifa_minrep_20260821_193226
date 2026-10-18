#!/usr/bin/env bash
set -u
W=/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017/seed40
source "$W/env.sh"
test ! -e "$ZR_RUN/pipeline.started" || exit 91
date -Is > "$ZR_RUN/pipeline.started"
echo $$ > "$ZR_RUN/pipeline.pid"
run_command() {
  label="$1"
  shift
  printf '%s START %s\n' "$(date -Is)" "$*" >> "$ZR_RUN/commands.log"
  "$@" > "$ZR_RUN/$label.log" 2>&1
  rc=$?
  printf '%s EXIT %s %s\n' "$(date -Is)" "$rc" "$label" >> "$ZR_RUN/commands.log"
  printf '%s\n' "$rc" > "$ZR_RUN/$label.exit"
  if [ "$rc" -ne 0 ]; then printf '%s\n' "$rc" > "$ZR_RUN/pipeline.exit"; exit "$rc"; fi
}
run_command train torchrun --standalone --nnodes=1 --nproc_per_node=1 -m zrg_exp.train
run_command evaluate python -u -m zrg_exp.evaluate
run_command finalize python -u -m zrg_exp.finalize
printf '0\n' > "$ZR_RUN/pipeline.exit"
