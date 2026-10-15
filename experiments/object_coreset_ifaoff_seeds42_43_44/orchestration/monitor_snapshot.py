import json,datetime,subprocess,shutil,statistics
from pathlib import Path
Q=Path('/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def text(p):return p.read_text()[-4000:] if p.exists() else None
r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue_exit':text(Q/'queue.exit'),'commands':text(Q/'commands.log'),
 'disk':shutil.disk_usage(Q)._asdict(),'gpu':subprocess.getoutput('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv'),'seeds':{}}
ps=subprocess.check_output(['ps','-eo','pid,ppid,pgid,etime,args'],text=True)
r['processes']='\n'.join(x for x in ps.splitlines() if any(k in x for k in ('original_protocol/queue.sh','original_protocol/seed_pipeline.sh',' -m coreset_exp.','preflight_noifa.py','spawn_main')) and 'monitor_snapshot.py' not in x)
for s in (42,43,44):
 root=Path(f'/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed{s}_20261014');v={'status':read(root/'RUN_STATUS.json'),'pipeline_exit':text(root/'pipeline.exit'),'commands':text(root/'commands.log'),'preflight':read(root/'GPU_PREFLIGHT.json')}
 for a in ('random','coreset'):
  p=root/a;ep=read(p/'epochs.json') or [];tr=[]
  for f in (p/'evaluation').glob('trials*.jsonl'):tr.extend(json.loads(l) for l in f.read_text().splitlines() if l.strip())
  v[a]={'epochs':len(ep),'steps':ep[-1]['steps'] if ep else 0,'last':ep[-1] if ep else None,'trials':len(tr),'training_complete':read(p/'TRAINING_COMPLETE.json'),'evaluation_complete':read(p/'evaluation/EVALUATION_COMPLETE.json')}
  if ep:
   speed=statistics.mean(x['seconds']/x['batches'] for x in ep[-10:]);v[a]['estimated_training_remaining_hours']=(105500-ep[-1]['steps'])*speed/3600
   v[a]['eta_basis']='Recent completed epoch seconds per optimizer step; storage/IFA startup overhead excluded'
 r['seeds'][s]=v
print(json.dumps(r,indent=2))
