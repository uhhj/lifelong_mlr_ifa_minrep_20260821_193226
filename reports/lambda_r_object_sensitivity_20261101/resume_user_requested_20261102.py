"""One explicit user-authorized SIGCONT of the retained process tree; no restart."""
import os,json,time,signal,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
E=Q/'engineering_device_snapshot_20261101_v1';D=Q/'user_resume_20261102_v1'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
assert not D.exists(),'Unique resume only; inspect existing receipt, never repeat'
p=js(Q/'user_pause_20261101_v1/STATUS.json');assert p['status']=='PAUSED_BY_USER_IN_MEMORY'
before={int(k):proc(int(k)) for k in p['after']}
for k,x in before.items():
 old=p['after'][str(k)]
 assert x and x['start']==old['start'] and x['cmd']==old['cmd'] and x['state']=='T',(k,x)
assert set(before)=={2163,2164,2169,3899,3900}
assert sha(Q/'CONTRACT.json')=='6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e'
assert sha(E/'CONTRACT.json')=='b9ebd306e7ac3ce1fba2ce107ba5b2fa6b4fd20f62164790bf7ced4897b6086c'
for c in [Q/'CONTRACT.json',E/'CONTRACT.json']:
 assert all(sha(Path(f))==h for f,h in js(c)['files'].items())
assert all(sha(Q/f)==h for f,h in p['light_hashes'].items()),'Paused light records changed'
for f in [Q/'controller.exit',E/'controller.exit',E/'FAILURE.json',E/'WATCHDOG_RELEASE.json',E/'HANDOFF_COMPLETE.json',Q/'L005_41/train_S.exit']:
 assert not f.exists(),str(f)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
assert gpu.returncode==0 and '2169' in gpu.stdout.split(),gpu.stderr
assert len(js(Q/'L005_41/EPOCHS_S.json'))==260
D.mkdir();actions=[]
def record(status,**extra):
 out=dict(status=status,time=time.time(),authorization='User 2026-11-02 explicitly requested continue training',before=before,actions=actions,pause_receipt_sha256=sha(Q/'user_pause_20261101_v1/STATUS.json'),source_hashes_verified=True,paused_light_hashes_verified=True,no_restart=True,no_checkpoint_reload=True,original_controller_remains_stopped=2163,**extra)
 t=D/'STATUS.tmp';t.write_text(json.dumps(out,indent=2));os.replace(t,D/'STATUS.json');return out
record('RESUME_IN_PROGRESS')
def cont(pid):
 x=proc(pid);assert x and x['start']==before[pid]['start'] and x['cmd']==before[pid]['cmd'] and x['state']=='T'
 os.kill(pid,signal.SIGCONT);actions.append(dict(pid=pid,signal='SIGCONT',time=time.time()));record('RESUME_IN_PROGRESS')
 for _ in range(100):
  x=proc(pid)
  if x and x['start']==before[pid]['start'] and x['state'] not in ['T','Z']:return
  time.sleep(.05)
 raise RuntimeError('Process failed to resume '+str(pid))
try:
 # Owner first, while watchdog remains stopped; require a fresh healthy wait heartbeat.
 oldtime=js(E/'STATUS.json')['time'];cont(3899)
 for _ in range(150):
  assert not (E/'FAILURE.json').exists()
  status=js(E/'STATUS.json')
  if status['time']>oldtime and status['phase']=='wait_L005_41':break
  time.sleep(.1)
 else:raise RuntimeError('No fresh owner heartbeat')
 cont(3900);cont(2164);cont(2169)
 time.sleep(3)
 after={k:proc(k) for k in before}
 assert after[2163]['state']=='T'
 assert all(after[k] and after[k]['start']==before[k]['start'] and after[k]['state'] not in ['T','Z'] for k in [3899,3900,2164,2169])
 assert not (E/'FAILURE.json').exists() and not (E/'WATCHDOG_RELEASE.json').exists()
 out=record('RESUMED_IN_MEMORY',after=after,train_status=js(Q/'L005_41/TRAIN_STATUS.json'),timing_excluded=dict(case='L005_41',stage=8,epoch=61,reason='Includes user-paused wall time; preserve raw record but exclude from throughput'),scientific_status='PARTIAL_NOT_ACCEPTED')
 print(json.dumps(out))
except BaseException as error:
 record('RESUME_INCOMPLETE_REQUIRES_AUDIT',error=repr(error),after={k:proc(k) for k in before});raise
