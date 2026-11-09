import os,json,time,signal,hashlib
from pathlib import Path
roots=[Path('/data/mlr_ifa/experiment_workspaces')/n for n in ['S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1','S_Q1_OBJECT41_42_43_20261109_rerun_v1','S_Q1_GOAL41_42_43_20261109_rerun_v1']]
D=roots[0]/'user_pause_20261109_v1'
expected={2638:'172343',2639:'172348',2641:'172352',2642:'172353',2650:'172477',2652:'172478'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
 allp={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   s=(p/'stat').read_text().split(') ',1)[1].split();allp[int(p.name)]=dict(pid=int(p.name),ppid=int(s[1]),state=s[0],start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
  except (OSError,ValueError):pass
 ids={pid for pid,x in allp.items() if pid!=os.getpid() and any(str(r) in x['cmd'] for r in roots)}
 while True:
  nxt=ids|{pid for pid,x in allp.items() if x['ppid'] in ids}
  if nxt==ids:break
  ids=nxt
 return {p:allp[p] for p in ids}
assert not D.exists(),'Existing pause receipt: inspect, never repeat'
assert sha(roots[0]/'CONTRACT.json')=='fbbce2917380d1cde1025f0006b6fdd6bc7959f92226bb5588cc431ee0c8d96c'
frozen=json.loads((roots[0]/'CONTRACT.json').read_text())['files']
assert all(sha(Path(p))==h for p,h in frozen.items())
before=inventory()
assert all(p in before and before[p]['start']==s for p,s in expected.items())
assert not any((r/'controller.exit').exists() for r in roots)
D.mkdir();actions=[]
def record(status,**extra):
 o=dict(status=status,observed=time.time(),authorization='User requested pause for storage expansion',before=before,actions=actions,automatic_resume_allowed=False,**extra)
 t=D/'STATUS.tmp';t.write_text(json.dumps(o,indent=2));os.replace(str(t),str(D/'STATUS.json'));return o
record('PAUSE_IN_PROGRESS')
def stop(x):
 now=inventory().get(x['pid'])
 if now is None:return
 assert now['start']==x['start'] and now['cmd']==x['cmd'],'Process identity changed'
 if now['state'] not in ['T','Z']:
  os.kill(x['pid'],signal.SIGSTOP);actions.append(dict(pid=x['pid'],start=x['start'],signal='SIGSTOP',observed=time.time()));record('PAUSE_IN_PROGRESS')
 for _ in range(100):
  now=inventory().get(x['pid'])
  if now is None or (now['start']==x['start'] and now['state'] in ['T','Z']):return
  time.sleep(.02)
 raise RuntimeError('Process failed to stop')
try:
 for pid in [2638,2639,2641,2642,2650,2652]:stop(before[pid])
 for _ in range(3):
  for x in inventory().values():stop(x)
  time.sleep(.2)
 after=inventory();assert all(x['state'] in ['T','Z'] for x in after.values())
 assert all(p in after and after[p]['start']==s and after[p]['state']=='T' for p,s in expected.items())
 assert all(sha(Path(p))==h for p,h in frozen.items())
 paths=[p for r in roots for p in r.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit'] and D not in p.parents]
 hashes={str(p):sha(p) for p in paths};time.sleep(2)
 assert all(sha(Path(p))==h for p,h in hashes.items()),'Paused records not stable'
 statuses={str(p):json.loads(p.read_text()) for r in roots[1:] for p in r.rglob('TRAIN_STATUS.json')}
 out=record('PAUSED_BY_USER_IN_MEMORY',after=after,source_files_verified=len(frozen),light_hashes=hashes,light_stable_seconds=2,training_statuses=statuses,checkpoint_claim='No forced checkpoint. Only previously saved complete work is durable; current partial work remains in memory.',expansion_boundary='Online expansion may preserve processes. Shutdown or reboot destroys live partial state; recovery audit required before any resume.',signals_used=['SIGSTOP'],optimizer_calls_added=0,environment_calls_added=0)
 print(json.dumps(out))
except BaseException as e:
 record('PAUSE_INCOMPLETE_REQUIRES_AUDIT',error=repr(e),after=inventory());raise
