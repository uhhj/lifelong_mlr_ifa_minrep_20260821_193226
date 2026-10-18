"""Lightweight true-state snapshot; never starts model or environment execution."""
import json,time,subprocess,hashlib,shutil
from pathlib import Path
Q=Path(__file__).resolve().parent
def txt(p):return p.read_text(errors='replace') if p.exists() else None
def js(p):return json.loads(p.read_text()) if p.exists() else None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(unix=time.time(),queue_status=js(Q/'RUN_STATUS.json'),queue_exit=txt(Q/'queue.exit'),wrapper_exit=txt(Q/'wrapper.exit'),commands=txt(Q/'commands.log'),seeds={})
out['orchestration_unchanged']=all(sha(Path(p))==h for p,h in (js(Q/'ORCHESTRATION_SHA256.json') or {}).items())
for s in (42,43,44):
 O=Path(f'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017');S=Q/f'seed{s}/source'
 c=js(O/'CONTRACT.json');d=dict(status=js(O/'RUN_STATUS.json'),pipeline_exit=txt(O/'pipeline.exit'),supervisor_exit=txt(O/'supervisor.exit'),preflight=js(O/'GPU_PREFLIGHT.json'),arms={})
 if c:
  d['source_unchanged']=all(sha(S/n)==h for n,h in c['source_sha256'].items())
  d['config_unchanged']=all(sha(Path(p))==h for p,h in c['configs'].items())
  d['modal_unchanged']=sha(O/'MODAL_CONTRACT.json')==c['modal_sha256']
 for arm in ('random','coreset'):
  r=O/arm;rows=js(r/'epochs.json') or []
  d['arms'][arm]=dict(epochs=len(rows),last=rows[-1] if rows else None,checkpoints=[p.name for p in (r/'models').glob('*.pth')],formal_records=sum(len(p.read_text().splitlines()) for p in (r/'evaluation').glob('trials*.jsonl')),attempt_events=sum(len(p.read_text().splitlines()) for p in (r/'evaluation').glob('attempts*.jsonl')))
  if rows:d['arms'][arm]['recent_epoch_seconds']=sum(x['seconds'] for x in rows[-10:])/len(rows[-10:])
 d['logs']={p.name:dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=p.read_text(errors='replace').splitlines()[-3:]) for p in O.glob('*.log')}
 out['seeds'][str(s)]=d
out['processes']=[x for x in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines() if ('Coreset_paper_GOAL' in x or ' -m coreset_exp.' in x or 'ZR_coreset_seeds40_41' in x) and 'monitor_snapshot' not in x and 'ps -eo' not in x]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
out['free_bytes']={p:shutil.disk_usage(p).free for p in ('/root','/data')}
p=Q/'monitor'/f'{time.time_ns()}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
(Q/'LATEST_SNAPSHOT.json').write_bytes(p.read_bytes());print(json.dumps(out,indent=2))
