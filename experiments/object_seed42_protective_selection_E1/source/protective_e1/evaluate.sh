#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
VIEW="${1:?primary or secondary}"
test -s "$E1_OUT/TRAINING_COMPLETE.json"
case "$VIEW" in primary|secondary) ;; *) exit 2;; esac
test ! -e "$E1_OUT/${VIEW}_COMPLETE.json" || exit 0
LOG_FILE="$E1_OUT/eval_${VIEW}.log"
export E1_TRIAL_LOG="$E1_OUT/trials_${VIEW}.jsonl"
test ! -e "$LOG_FILE" && test ! -e "$E1_TRIAL_LOG" || { echo 'Partial evaluation retained; explicit technical recovery required'; exit 3; }
ARGS=(seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic
 folder=/data/mlr_ifa/datasets bddl_folder="$E1_SOURCE/libero/libero/bddl_files"
 init_states_folder="$E1_SOURCE/libero/libero/init_files"
 pretrain=/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth
 eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true
 +epoch_dir="$E1_OUT/run_001" +test_log_file="$LOG_FILE" training=false use_wandb=false)
if [[ "$VIEW" == primary ]]; then
 python -m protective_e1.evaluate "${ARGS[@]}" '+eval_epochs=[99]' +experiment_root="$E1_OUT/eval_primary"
else
 python -m protective_e1.evaluate "${ARGS[@]}" +eval_current_only=true '+eval_epochs=[70,75,80,85,90,95,99]' +experiment_root="$E1_OUT/eval_secondary_selection"
 read -r E6 E7 E8 E9 < <(python - "$LOG_FILE" <<'PY'
import re,sys
p=re.compile(r'\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)');r={}
for line in open(sys.argv[1]):
 m=p.search(line)
 if m:
  t,e,s=map(int,m.group(1,2,3));r[s,e,t]=float(m[4])
print(*(max((70,75,80,85,90,95,99),key=lambda e:(r[s,e,s],e)) for s in range(6,10)))
PY
 )
 for SPEC in "7:$E7:[6]" "8:$E8:[6,7]" "9:$E9:[6,7,8]"; do
  IFS=: read -r STAGE EPOCH TASKS <<< "$SPEC"
  python -m protective_e1.evaluate "${ARGS[@]}" +eval_stage="$STAGE" +eval_task_ids="$TASKS" +eval_epochs="[$EPOCH]" +experiment_root="$E1_OUT/eval_secondary_stage${STAGE}"
 done
fi
python - "$VIEW" <<'PY'
import os,json,sys,re
from pathlib import Path
from protective_e1.io import write,sha
o=Path(os.environ['E1_OUT']);v=sys.argv[1]
rows=[json.loads(x) for x in (o/('trials_'+v+'.jsonl')).read_text().splitlines()]
n=200 if v=='primary' else 680
assert len(rows)==n
assert all(type(r['success']) is bool for r in rows)
write(o/(v+'_COMPLETE.json'),dict(status='PASS',records=n,trials_sha256=sha(o/('trials_'+v+'.jsonl'))))
PY
