#!/usr/bin/env bash
set -euo pipefail
W=/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed42
source "$W/env.sh"
phase() {
  name="$1"; shift
  test ! -e "$W/$name.exit"
  printf '%s\n' "$*" >> "$W/admission_commands.log"
  set +e
  "$@" > "$W/$name.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$W/$name.exit"
  tail -n 12 "$W/$name.log"
  test "$rc" -eq 0
}
phase software /data/mlr_ifa/env/bin/python -m unittest ra_exp.test_reference -v
phase prepare /data/mlr_ifa/env/bin/python -m ra_exp.prepare
phase offline /data/mlr_ifa/env/bin/python -u -m ra_exp.select_offline
phase freeze /data/mlr_ifa/env/bin/python -m ra_exp.freeze
phase preflight /data/mlr_ifa/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 -m ra_exp.preflight
echo ADMISSION_PASS_NOT_LAUNCHED
