#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec 9>"$E1_OUT/process.lock"
flock -n 9
printf '%s\n' "$$" > "$E1_OUT/pipeline.pid"
trap 'code=$?; printf "%s\n" "$code" > "$E1_OUT/pipeline.exit"' EXIT
run_step() {
 local label="$1"; shift
 printf '%s START %s\n' "$(date -Is)" "$label" >> "$E1_OUT/commands.log"
 "$@" > "$E1_OUT/${label}.log" 2>&1
 printf '%s EXIT 0 %s\n' "$(date -Is)" "$label" >> "$E1_OUT/commands.log"
}
run_step prepare python -B -m protective_e1.prepare
if [[ ! -e "$E1_OUT/REAL_PREFLIGHT.json" ]]; then
 run_step preflight python -m torch.distributed.run --nproc_per_node=1 --master_port=37984 -m protective_e1.preflight
fi
run_step train python -m torch.distributed.run --nproc_per_node=1 --master_port=37985 -m protective_e1.train
run_step evaluate_primary bash protective_e1/evaluate.sh primary
run_step evaluate_secondary bash protective_e1/evaluate.sh secondary
run_step finalize python -B -m protective_e1.finalize
