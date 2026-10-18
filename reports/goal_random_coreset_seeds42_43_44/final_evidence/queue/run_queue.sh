#!/usr/bin/env bash
set -euo pipefail
Q=/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017
exec 8>"$Q/queue.lock"
flock -n 8 || exit 70
test ! -e "$Q/queue.started"
date -Is > "$Q/queue.started"
echo $$ > "$Q/queue.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$Q/queue.exit"' EXIT
run() {
 label="$1"; shift
 printf '%s START %s ' "$(date -Is)" "$label" >> "$Q/commands.log"
 printf '%q ' "$@" >> "$Q/commands.log"
 printf '\n' >> "$Q/commands.log"
 set +e
 "$@" > "$Q/$label.log" 2>&1
 rc=$?
 set -e
 printf '%s EXIT %s %s\n' "$(date -Is)" "$label" "$rc" >> "$Q/commands.log"
 test "$rc" = 0 || exit "$rc"
}
run wait_predecessor /data/mlr_ifa/env/bin/python -u "$Q/admission.py"
for seed in 42 43 44; do
 source "$Q/seed${seed}/source/coreset_exp/env.sh"
 test ! -e "$CORESET_OUT/pipeline.started"
 run "configure_$seed" python -u -m coreset_exp.configure
 run "suite_audit_$seed" python -u "$Q/audit_suite.py"
 # Software tests were already run before queue freeze; verify rather than repeat.
 test "$(cat "$Q/software_${seed}.exit")" = 0
 run "preflight_$seed" torchrun --nproc_per_node=1 --master_port=$((39342+seed-42)) -m coreset_exp.preflight
 run "supervise_$seed" bash "$CORESET_SOURCE/coreset_exp/supervise.sh"
 test "$(cat "$CORESET_OUT/pipeline.exit")" = 0
 test "$(cat "$CORESET_OUT/supervisor.exit")" = 0
done
run aggregate python -u "$Q/aggregate_three_seeds.py"
