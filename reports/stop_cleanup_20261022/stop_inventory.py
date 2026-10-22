import os,json,signal,time,subprocess,hashlib
from pathlib import Path
ROOT=Path('/data/mlr_ifa')
def processes():
 out={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve());st=(p/'stat').read_text().split(') ',1)[1].split()
   if cmd and int(p.name)!=os.getpid() and 'python -c' not in cmd:out[int(p.name)]=dict(command=cmd,cwd=cwd,ppid=int(st[1]),state=st[0])
  except (OSError,ValueError):pass
 return out
def relevant(v):return '/data/mlr_ifa/experiment_workspaces/' in v['command'] or v['cwd'].startswith('/data/mlr_ifa/experiment_workspaces/')
ps=processes();roots={p for p,v in ps.items() if relevant(v)};controllers={p for p in roots if 'controller' in ps[p]['command'] or 'capacity_recovery' in ps[p]['command']}
for p in controllers:os.kill(p,signal.SIGSTOP)
ps=processes();targets={p for p,v in ps.items() if relevant(v)}
while True:
 new=targets|{p for p,v in ps.items() if v['ppid'] in targets}
 if new==targets:break
 targets=new
before={str(p):ps[p] for p in targets}
for p in controllers:
 try:os.kill(p,signal.SIGTERM);os.kill(p,signal.SIGCONT)
 except ProcessLookupError:pass
for p in sorted(targets-controllers,reverse=True):
 try:os.kill(p,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(3);left=processes();forced=[]
for p in targets:
 if p in left and left[p]['state']!='Z':
  assert left[p]['command']==ps[p]['command'];os.kill(p,signal.SIGKILL);forced.append(p)
time.sleep(1);left=processes();remaining={str(p):v for p,v in left.items() if relevant(v) and v['state']!='Z'};assert not remaining,remaining
disk=lambda:{m:dict(ordinary=os.statvfs(m).f_bavail*os.statvfs(m).f_frsize,root=os.statvfs(m).f_bfree*os.statvfs(m).f_frsize) for m in ('/','/data')}
receipt=dict(status='USER_STOPPED_ALL_CURRENT_TRAINING_AND_DISPATCH',unix=time.time(),before=before,forced=forced,remaining=remaining,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),disk=disk(),runs={})
for o in (ROOT/'experiments').glob('CAETv2_*_20261022'):
 ep=o/'ra_coreset/epochs.json';h=json.loads(ep.read_text()) if ep.exists() else []
 receipt['runs'][str(o)]=dict(complete_epochs=len(h),last_epoch=h[-1] if h else None,formal_records=sum(len(p.read_text().splitlines()) for p in (o/'ra_coreset/evaluation').glob('trials*.jsonl')))
q=ROOT/'experiment_workspaces/CAETv2_TAC_AUX8_OBJECT41_20261022';p=q/'USER_STOP_ALL_TRAINING.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n')
inventory={}
for base in ('/data/mlr_ifa','/root/mlr_ifa','/data/mlr_ifa/experiments','/data/mlr_ifa/experiment_workspaces','/data/mlr_ifa/tmp','/root/.cache'):
 if Path(base).exists():inventory[base]=subprocess.check_output(['du','-x','-B1','--max-depth=1',base],text=True,stderr=subprocess.DEVNULL)
receipt['inventory']=inventory
print(json.dumps(receipt))
