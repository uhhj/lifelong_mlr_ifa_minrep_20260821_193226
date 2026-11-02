"""Run the unchanged evaluator under the existing collection/environment mutex."""
import os,sys,json,time,fcntl,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;case=sys.argv[1]
assert case in [g+'_'+str(k) for g in ['L005','L020'] for k in [41,42,43]]
lock=Path('/data/mlr_ifa/'+Q.name+'_collection.lock').open('a')
def event(kind):
 with (E/(case+'_EVALUATION_LOCK.jsonl')).open('a') as f:f.write(json.dumps(dict(event=kind,time=time.time(),case=case,pid=os.getpid(),source='parent_lock_unchanged_evaluator'))+'\n')
fcntl.flock(lock,fcntl.LOCK_EX);event('acquired')
try:rc=subprocess.run(['/data/mlr_ifa/env/bin/python','-B',str(Q/case/'evaluate.py')],cwd=Q/case,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')).returncode
finally:event('released');fcntl.flock(lock,fcntl.LOCK_UN)
sys.exit(rc)
