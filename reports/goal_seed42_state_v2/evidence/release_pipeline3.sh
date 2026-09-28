#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/activate.sh
base=/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2
cd "$base/source"
export PYTHONPATH="$PWD:$PYTHONPATH" CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(--pretrained /data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth --data-root /data/mlr_ifa/datasets --bddl-root /data/mlr_ifa/src/libero/libero/bddl_files --init-states-root /data/mlr_ifa/src/libero/libero/init_files)
bash scripts/run_goal_seed42.sh --mode preflight "${args[@]}" --run-dir "$base/run"
cp "$base/evidence/protocol_audit.md" "$base/run/protocol_audit.md"
cp "$base/evidence/git_status.txt" "$base/run/git_status.txt"
bash scripts/run_goal_seed42.sh --mode smoke "${args[@]}" --run-dir "$base/smoke_v2_release3" --smoke-dir "$base/smoke_v2_release3"
python scripts/test_goal_buffer_and_updates.py --run-dir "$base/smoke_v2_release3" > "$base/smoke_v2_release3/logs/additional.log" 2>&1
python - <<'PY'
import json
from pathlib import Path
from libero.lifelong.runbook_core import atomic_json
root=Path('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/smoke_v2_release3')
r=json.loads((root/'smoke/additional/result.json').read_text())
assert r['status']=='PASS' and r['workers_0_vs_12_same_order_and_parent_torch_rng']
g=json.loads((root/'test_gate.json').read_text());g['additional_checks']=r
atomic_json(root/'test_gate.json',g)
print('RELEASE_GATE_PASS',flush=True)
PY
bash "$base/evidence/full_run.sh"
