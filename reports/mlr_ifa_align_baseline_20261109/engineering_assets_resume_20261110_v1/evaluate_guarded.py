"""Run unchanged evaluator; verify pre-cell RNG against preserved failed attempt."""
from resume_runtime import *
import fcntl,runpy
case=sys.argv[1];assert case in CASES
lock=(Q/'environment.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
event('environment_acquired',case=case,pid=os.getpid());code=1
try:
 sys.path.insert(0,str(Q/'cases'/case/'source'))
 if case=='OBJECT_41':
  import torch,numpy as np
  from coreset_exp import common
  oldsave=common.save
  oldrng=torch.load(E/'failed_evaluation_OBJECT41/rng_before_stage6_task6.pth',map_location='cpu')
  def equal(a,b):
   if torch.is_tensor(a):return torch.equal(a,b)
   if isinstance(a,np.ndarray):return np.array_equal(a,b)
   if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
   if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
   return a==b
  def guarded_save(path,obj):
   if Path(path).name=='rng_before_stage6_task6.pth':
    assert equal(oldrng,obj),'Pre-cell RNG changed; stop before constructing any environment'
    write(E/'RNG_BOUNDARY_AUDIT.json',dict(status='PASS_EXACT_PRE_CELL_RNG',case=case,python=True,numpy=True,cpu=True,cuda=True,old_sha256=sha(E/'failed_evaluation_OBJECT41/rng_before_stage6_task6.pth'),observed=time.time(),new_optimizer=0,new_environment_before_guard=0))
   return oldsave(path,obj)
  common.save=guarded_save
 sys.argv=['coreset_exp.evaluate','random']
 runpy.run_module('coreset_exp.evaluate',run_name='__main__')
 code=0
finally:
 event('environment_released',case=case,pid=os.getpid(),code=code)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
