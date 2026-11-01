"""Explicit user-requested in-memory pause. No termination, restart or resume."""
import os,json,time,signal,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
E=Q/'engineering_device_snapshot_20261101_v1';D=Q/'user_pause_20261101_v1'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
def inventory():
 out={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   x=proc(int(p.name))
   if x and x['pid']!=os.getpid() and str(Q) in x['cmd']:out[x['pid']]=x
  except (OSError,ValueError):pass
 return out
assert not D.exists(),'Single pause request; inspect existing receipt instead of repeating'
assert not (Q/'controller.exit').exists() and not (E/'controller.exit').exists()
assert sha(Q/'CONTRACT.json')=='6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e'
assert sha(E/'CONTRACT.json')=='b9ebd306e7ac3ce1fba2ce107ba5b2fa6b4fd20f62164790bf7ced4897b6086c'
g=js(E/'GUARD.json');owner=js(E/'STARTED.json')['identity'];watch=js(E/'WATCHDOG_READY.json')['pid']
before=inventory();assert owner['pid'] in before and before[owner['pid']]['start']==owner['start']
assert watch in before and str(E/'watchdog.py') in before[watch]['cmd']
assert not (E/'HANDOFF_COMPLETE.json').exists(),'Actual boundary changed; inspect before adapting pause'
for x in [g['controller'],g['child'],g['training']]:
 assert x['pid'] in before and before[x['pid']]['start']==x['start']
D.mkdir();actions=[]
def record(status,**extra):
 out=dict(status=status,time=time.time(),before=before,actions=actions,authorization='User explicitly requested pause training',mechanism='SIGSTOP verified processes, retain in-memory partial work; never terminate/restart',**extra)
 tmp=D/'STATUS.tmp';tmp.write_text(json.dumps(out,indent=2));os.replace(tmp,D/'STATUS.json');return out
record('PAUSE_IN_PROGRESS')
def stop(x):
 live=proc(x['pid']);assert live and live['start']==x['start'] and live['cmd']==x['cmd']
 if live['state']=='Z':return
 if live['state']!='T':
  os.kill(x['pid'],signal.SIGSTOP);actions.append(dict(pid=x['pid'],start=x['start'],signal='SIGSTOP',time=time.time()))
 for _ in range(100):
  live=proc(x['pid'])
  if live and live['start']==x['start'] and live['state'] in ['T','Z']:return
  time.sleep(.02)
 raise RuntimeError('Target not stopped: '+str(x['pid']))
try:
 # Stop watchdog before owner so it cannot resume the old coordinator.
 for pid in [watch,owner['pid'],g['controller']['pid'],g['child']['pid'],g['training']['pid']]:stop(before[pid]);record('PAUSE_IN_PROGRESS')
 for _ in range(3):
  for x in inventory().values():stop(x)
  time.sleep(.2)
 after=inventory();assert after and all(x['state'] in ['T','Z'] for x in after.values())
 assert all(sha(Path(p))==h for contract in [Q/'CONTRACT.json',E/'CONTRACT.json'] for p,h in js(contract)['files'].items())
 paths=[p for p in Q.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit'] and D not in p.parents]
 hashes={str(p.relative_to(Q)):sha(p) for p in paths};time.sleep(2)
 assert all(sha(Q/p)==h for p,h in hashes.items()),'Paused records changed'
 GPU=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
 out=record('PAUSED_BY_USER_IN_MEMORY',after=after,source_hashes_verified=True,light_hashes=hashes,light_stable_seconds=2,train_status=js(Q/'L005_41/TRAIN_STATUS.json'),queue_status=js(Q/'STATUS.json'),gpu=dict(exit=GPU.returncode,stdout=GPU.stdout,stderr=GPU.stderr),checkpoint_claim='Only last fully saved epoch is on disk; current partial block remains in live memory. No new checkpoint forced.',partial_epoch_cost='Not inferred as zero; in-memory counters retained but not read.',resume_allowed=False,automatic_handoff_paused=True)
 print(json.dumps(out))
except BaseException as error:
 record('PAUSE_INCOMPLETE_REQUIRES_AUDIT',error=repr(error),after=inventory());raise
