"""Serialize exact original collect/evaluate subprocesses; no RNG/model operations."""
import fcntl,json,os,subprocess,sys,time,uuid
from pathlib import Path
Q=Path(__file__).resolve().parent
case,kind=sys.argv[1:3]
assert case in ['Q1_41', 'Q1_42', 'Q1_43']
assert kind in ['collect','evaluate']
extra=sys.argv[3:]
assert (kind=='collect' and len(extra)==2 and extra[0]=='S' and extra[1] in ['6','7','8','9']) or (kind=='evaluate' and not extra)
event_id=str(uuid.uuid4()); lock_path=Path('/data/mlr_ifa/S_Q1_GOAL41_42_43_20261108_v1_environment.lock')
def event(phase,**kw):
 with (Q/'ENVIRONMENT_LOCK_EVENTS.jsonl').open('a') as f:
  f.write(json.dumps(dict(event_id=event_id,phase=phase,case=case,kind=kind,args=extra,pid=os.getpid(),time=time.time(),**kw))+'\n');f.flush();os.fsync(f.fileno())
with lock_path.open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 event('acquired')
 code=-1
 try:
  code=subprocess.call([sys.executable,'-B',str(Q/case/(kind+'.py'))]+extra,cwd=Q/case)
 finally:event('released',returncode=code)
sys.exit(code)
