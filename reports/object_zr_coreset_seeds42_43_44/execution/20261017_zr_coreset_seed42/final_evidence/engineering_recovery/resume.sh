#!/usr/bin/env bash
set -u
W=/data/mlr_ifa/experiment_workspaces/ZR_coreset_OBJECT_seed42_20261017
E="$W/engineering_recovery"
source "$W/env.sh"
test ! -e "$E/resume.started" || exit 91
date -Is > "$E/resume.started"
echo $$ > "$E/resume.pid"
run_command() {
  label="$1"
  shift
  printf '%s START %s\n' "$(date -Is)" "$*" >> "$E/commands.log"
  "$@" > "$E/$label.log" 2>&1
  rc=$?
  printf '%s EXIT %s %s\n' "$(date -Is)" "$rc" "$label" >> "$E/commands.log"
  printf '%s\n' "$rc" > "$E/$label.exit"
  if [ "$rc" -ne 0 ]; then printf '%s\n' "$rc" > "$E/resume.exit"; exit "$rc"; fi
}
run_command train torchrun --standalone --nnodes=1 --nproc_per_node=1 "$E/train_metadata_adapter.py"
run_command evaluate python -u -m zrg_exp.evaluate
run_command finalize python -u -m zrg_exp.finalize
printf '0\n' > "$E/resume.exit"
