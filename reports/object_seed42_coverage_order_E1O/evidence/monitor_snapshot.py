"""Read-only E1-O status; never launches or loads model tensors."""
import csv,hashlib,json,os,subprocess,time
from pathlib import Path
O=Path('/data/mlr_ifa/experiments/E1O_coverage_order_seed42_20261009')
S=Path('/data/mlr_ifa/experiment_workspaces/E1O_coverage_order_20261009/source')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contract=read(O/'CONTRACT.json')
source={k:sha(S/k)==v for k,v in contract['source_sha256'].items()}
curve=O/'run_001/paper_repro_train_curve.csv'
rows=list(csv.DictReader(curve.open())) if curve.exists() else []
episodes={v:sum(1 for _ in (O/('trials_'+v+'.jsonl')).open()) if (O/('trials_'+v+'.jsonl')).exists() else 0 for v in ['primary','secondary']}
logs={p.name:dict(size=p.stat().st_size,modified_unix=p.stat().st_mtime) for p in O.glob('*.log')}
processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
live=[p for p in processes if any(s in p for s in ['-m protective_e1.train','-m protective_e1.evaluate','protective_e1/pipeline.sh']) and 'bash -c' not in p]
latest_order=None
if rows:
    last=rows[-1];p=O/'order_logs'/('task'+last['stage'])/('epoch%03d.json'%int(last['epoch']))
    d=read(p)
    if d:latest_order=dict(stage=d['stage'],epoch=d['epoch'],batches=len(d['batches']),all_ids_once=d['all_ids_once'],schedules=len(d['schedules']),actual_slots=len(d['actual_ids']))
result=dict(unix=time.time(),status=read(O/'RUN_STATUS.json'),contract_sha256=sha(O/'CONTRACT.json'),
    frozen_source_files=len(source),frozen_source_unchanged=all(source.values()),source_mismatches=[k for k,v in source.items() if not v],
    complete_epochs=len(rows),complete_optimizer_steps=sum(int(r['batches']) for r in rows),latest_epoch=rows[-1] if rows else None,
    latest_order=latest_order,formal_episodes=episodes,formal_checkpoints=len(list((O/'run_001').glob('mlrifa_*_model_ep*.pth'))),
    recovery_exists=(O/'recovery/latest.pth').exists(),pipeline_exit=(O/'pipeline.exit').read_text().strip() if (O/'pipeline.exit').exists() else None,
    processes=live,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
    logs=logs,commands=(O/'commands.log').read_text() if (O/'commands.log').exists() else '')
print(json.dumps(result,indent=2))
