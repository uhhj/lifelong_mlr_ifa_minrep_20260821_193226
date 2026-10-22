"""Read-only frozen queue/process/GPU/disk/epoch/attempt census."""
import os,json,time,hashlib,subprocess,math
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
def js(p):return json.loads(Path(p).read_text())
def read(p):return js(p) if p.exists() and p.suffix=='.json' else p.read_text() if p.exists() else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=js(Q/'ORCHESTRATION_CONTRACT.json')
out=dict(observed_unix=time.time(),queue=str(Q),frozen_hashes=dict(count=len(c['files']),failures=[p for p,h in c['files'].items() if sha(p)!=h],contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json')),queue_state={n:read(Q/n) for n in ['RUN_STATUS.json','controller.pid.json','controller.exit','FAILURE.json','TACR_GATE.json','TACRST_GATE.json','TACR_EXPANSION_NOT_STARTED.json','TACRST_EXPANSION_NOT_STARTED.json','PRIOR_PREFIX_ACCEPTANCE.json','DISPATCH_RECEIPT.json','CAPACITY_PLAN.json']},runs={},processes=[])
for e in js(Q/'QUEUE.json'):
 W=Q/e['key'];O=Path(e['output']);r=dict(entry=e,exists=O.exists(),worker={n:read(W/n) for n in ['supervisor.pid.json','supervisor.exit','FAILURE.json']},admission={n:read(W/'admission'/n) for n in ['prepare.exit','labels6.exit','gpu_preflight.exit','pipeline.exit','final_process.exit']},receipt=read(Q/(e['key']+'.receipt.json')))
 if O.exists():
  r['status_files']={n:read(O/n) for n in ['RUN_STATUS.json','GPU_PREFLIGHT.json','TASK6_TENSOR_PARITY.json','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','pipeline.exit','supervisor.exit']};r['phase_exits']={p.name:p.read_text() for p in O.glob('*.exit')}
  if (O/'CONTRACT.json').exists():
   co=js(O/'CONTRACT.json');r['input_hashes']=dict(count=len(co['files']),failures=[p for p,h in co['files'].items() if sha(p)!=h])
  history=read(O/'ra_coreset/epochs.json') or [];r['epochs']=history;r['ledger_by_epoch']=[]
  for h in history:
   p=O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(h['task'],h['epoch']))
   if not p.exists():r['ledger_by_epoch'].append(dict(task=h['task'],epoch=h['epoch'],pending_atomic_write=True));continue
   ss=js(p);n=e['spec']['dataset_sizes'][str(h['task'])]
   r['ledger_by_epoch'].append(dict(task=h['task'],epoch=h['epoch'],records=len(ss),steps=[s['step'] for s in ss],ids_complete=sorted(i for s in ss for i in s['batch_ids'])==list(range(n)),single_main_and_rep=all(s['committed_updates']==s['representation_optimizer_calls']==s['head_updates']==s['optimizer_trials']==1 and s['variant']==e['variant'] for s in ss),bc_valid=sum(s['bc_valid'] for s in ss),bc_invalid=sum(s['bc_invalid'] for s in ss),effects=sum(s['effect_samples'] for s in ss),anchors=sum(s['cross_anchors'] for s in ss),key_reads=sum(s['old_key_reads'] for s in ss),residual_ratio_max=max(s['residual_ratio_max'] for s in ss)))
  E=O/'ra_coreset/evaluation'
  def rows(pattern):return [json.loads(l) for p in sorted(E.glob(pattern)) for l in p.read_text().splitlines()]
  r['formal']=dict(trials=rows('trials*.jsonl'),attempts=rows('attempts*.jsonl'),cells={p.name:js(p) for p in E.glob('cell*.json')})
  r['logs']={p.name:p.read_text(errors='replace')[-4000:] for p in list(O.glob('*.log'))+list((W/'admission').glob('*.log')) if p.stat().st_size}
 out['runs'][e['key']]=r
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(Q) in cmd and 'python -c' not in cmd:out['processes'].append(dict(pid=int(p.name),command=cmd))
 except OSError:pass
out['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
out['disk']={m:dict(ordinary=(v:=os.statvfs(m)).f_bavail*v.f_frsize,root=v.f_bfree*v.f_frsize) for m in ('/','/data')}
print(json.dumps(out))
