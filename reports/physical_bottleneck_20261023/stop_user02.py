"""Stop only the independently identified canceled experiment; retain all assets."""
import os,signal,time,json,hashlib,datetime
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_BOTTLENECK_OBJECT41_20261023_v1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def processes():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   env=(p/'environ').read_bytes().split(b'\0')
   if ('PBD_ROOT='+str(Q)).encode() in env:
    rows.append(dict(pid=int(p.name),pgid=os.getpgid(int(p.name)),sid=os.getsid(int(p.name)),cmd=cmd))
  except (OSError,ProcessLookupError):pass
 return rows
before=processes();events=[]
if before:
 ids={r['sid'] for r in before};assert ids=={120819},before
 leader=next((r for r in before if r['pid']==120819),None)
 assert leader and 'controller.py' in leader['cmd'] and leader['pgid']==120819,before
 os.killpg(120819,signal.SIGSTOP);events.append('SIGSTOP confirmed session 120819')
 os.killpg(120819,signal.SIGTERM);os.killpg(120819,signal.SIGCONT);events.append('SIGTERM then SIGCONT confirmed session')
 for _ in range(100):
  if not processes():break
  time.sleep(.1)
 if processes():
  assert all(r['sid']==120819 and r['pgid']==120819 for r in processes())
  os.killpg(120819,signal.SIGKILL);events.append('SIGKILL confirmed remaining group');time.sleep(1)
after=processes();assert not after,after
files={str(p.relative_to(Q)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in Q.rglob('*.pth')}
exits={str(p.relative_to(Q)):p.read_text() for p in Q.glob('*.exit')}
stamp=datetime.datetime.now().isoformat()
result=dict(status='EARLY_TERMINATED_BY_USER',time=stamp,root=str(Q),before=before,signals=events,remaining=after,retained_pth=files,original_exits=exits,old_status=json.loads((Q/'output/RUN_STATUS.json').read_text()),no_assets_deleted=True)
out=Q/('user_termination_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();(out/'RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
