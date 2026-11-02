"""Exact-path startup lock, before torch/environment imports; parent process untouched."""
import os,sys,json,time,fcntl,atexit,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent
c=json.loads((E/'CONTRACT.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['hook_files'].items())
config=json.loads((E/'HOOK_CONFIG.json').read_text());target=str(Path(sys.argv[0]).resolve())
case=config['targets'][target];assert sys.argv[1]=='S' and int(sys.argv[2]) in [6,7,8,9]
stage=int(sys.argv[2]);assert (E/'HOOK_ENABLED.json').exists()
assert not hasattr(sys,'_mlr_collection_lock_handle'),'No duplicate startup hook'
started=time.time();handle=Path(config['lock_path']).open('a');fcntl.flock(handle,fcntl.LOCK_EX)
sys._mlr_collection_lock_handle=handle
def event(kind):
 with (E/(case+'_COLLECTION_LOCK.jsonl')).open('a') as f:
  f.write(json.dumps(dict(event=kind,time=time.time(),stage=stage,pid=os.getpid(),case=case,wait_seconds=time.time()-started,source='pre_import_startup_hook'))+'\n');f.flush()
event('acquired')
def release():
 event('released');handle.close()
atexit.register(release)
