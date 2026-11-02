"""UNDEPLOYED: existing exact update path, with serialized training collection."""
import os,sys,json,hashlib,fcntl,subprocess,time
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PREV=Q/'engineering_checks_3090_20261102_v1';case=sys.argv[1]
assert case in ['L005_43','L020_41','L020_42','L020_43']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((E/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
assert json.loads((E/'PARALLEL_ADMISSION.json').read_text())['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
assert (E/'HANDOFF_COMPLETE.json').exists()
original_run=subprocess.run
def serialized_collection(argv,*args,**kwargs):
 if isinstance(argv,list) and len(argv)>2 and argv[2]==str(Q/case/'collect.py'):
  started=time.time()
  with Path('/data/mlr_ifa/'+Q.name+'_collection.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   with (E/(case+'_COLLECTION_LOCK.jsonl')).open('a') as f:f.write(json.dumps(dict(event='acquired',time=time.time(),wait_seconds=time.time()-started,stage=argv[-1],pid=os.getpid()))+'\n')
   result=original_run(argv,*args,**kwargs)
   with (E/(case+'_COLLECTION_LOCK.jsonl')).open('a') as f:f.write(json.dumps(dict(event='released',time=time.time(),stage=argv[-1],returncode=result.returncode,pid=os.getpid()))+'\n')
   return result
 return original_run(argv,*args,**kwargs)
subprocess.run=serialized_collection
scope=dict(__file__=str(PREV/'train_wrapper.py'),__name__='__main__')
exec(compile((PREV/'train_wrapper.py').read_text(),str(PREV/'train_wrapper.py'),'exec'),scope)
