"""Host read-only snapshot. No model forward, optimizer, CUDA allocation, or writes."""
import os,json,time,hashlib,subprocess,math,collections
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022')
def js(p):return json.loads(Path(p).read_text())
def read(p):return js(p) if p.exists() and p.suffix=='.json' else p.read_text() if p.exists() else None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
out=dict(observed_unix=time.time(),queue=str(Q),queue_state={n:read(Q/n) for n in ['RUN_STATUS.json','controller.pid.json','controller.exit','FAILURE.json','OBJECT_GATE.json','GOAL_NOT_STARTED.json','GOAL_AGGREGATE.json','DISPATCH_RECEIPT.json','CAPACITY_PLAN.json']},processes=[],runs={})
c=js(Q/'ORCHESTRATION_CONTRACT.json');out['frozen_hashes']=dict(count=len(c['files']),failures=[p for p,h in c['files'].items() if sha(p)!=h],contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'))
for e in js(Q/'QUEUE.json'):
 W=Q/e['key'];O=Path(e['output']);r=dict(entry=e,exists=O.exists(),worker={n:read(W/n) for n in ['supervisor.pid.json','supervisor.exit','FAILURE.json']},admission={n:read(W/'admission'/n) for n in ['prepare.exit','labels6.exit','gpu_preflight.exit','pipeline.exit','final_process.exit']},receipt=read(Q/(e['key']+'.receipt.json')),independent=read(Q/'independent'/(e['key']+'.json')))
 if O.exists():
  r['status_files']={n:read(O/n) for n in ['RUN_STATUS.json','GPU_PREFLIGHT.json','TASK6_TENSOR_PARITY.json','pipeline.exit','supervisor.exit','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json']}
  history=read(O/'ra_coreset/epochs.json') or [];r['epochs']=history
  r['input_hashes']=dict(count=0,failures=[])
  if (O/'CONTRACT.json').exists():
   co=js(O/'CONTRACT.json');r['input_hashes']=dict(count=len(co['files']),failures=[p for p,h in co['files'].items() if sha(p)!=h])
  r['initial_matches_parent']=js(O/'ra_coreset/INITIAL_STATE.json')==js(Path(e['parent'])/'ra_coreset/INITIAL_STATE.json') if (O/'ra_coreset/INITIAL_STATE.json').exists() else None
  r['ledger_by_epoch']=[]
  for h in history:
   p=O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(h['task'],h['epoch']))
   if not p.exists():r['ledger_by_epoch'].append(dict(task=h['task'],epoch=h['epoch'],pending_atomic_write=True));continue
   stats=js(p);n=e['spec']['dataset_sizes'][str(h['task'])]
   valid=all(s['variant']=='ALIGN' and s['committed_updates']==s['optimizer_trials']==1 and not any(s.get(k,0) for k in ['effect_samples','cross_anchors','head_updates','extra_partner_forwards']) for s in stats)
   r['ledger_by_epoch'].append(dict(task=h['task'],epoch=h['epoch'],records=len(stats),steps=[s['step'] for s in stats],ids_complete=sorted(i for s in stats for i in s['batch_ids'])==list(range(n)),native_only=valid,bc_valid=sum(s['bc_valid'] for s in stats),bc_invalid=sum(s['bc_invalid'] for s in stats)))
  E=O/'ra_coreset/evaluation'
  def rows(pattern):return [json.loads(line) for f in sorted(E.glob(pattern)) for line in f.read_text().splitlines()]
  r['formal']=dict(trials=rows('trials*.jsonl'),attempts=rows('attempts*.jsonl'),cells={p.name:js(p) for p in E.glob('cell*.json')})
  r['logs']={p.name:p.read_text(errors='replace')[-5000:] for p in list(O.glob('*.log'))+list((W/'admission').glob('*.log')) if p.stat().st_size}
 out['runs'][e['key']]=r
out['prefix_independent']=read(Q/'independent/OBJECT_41.json')
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
out['disk']={m:dict(ordinary=(v:=os.statvfs(m)).f_bavail*v.f_frsize,root=v.f_bfree*v.f_frsize) for m in ('/data','/')}
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd and str(Q) in cmd+cwd:out['processes'].append(dict(pid=int(p.name),command=cmd,cwd=cwd))
 except OSError:pass
print(json.dumps(out))
