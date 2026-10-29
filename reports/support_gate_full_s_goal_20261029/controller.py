"""Finite single dispatch: both complete S training sequences, then both evaluations."""
import os,subprocess,json,time,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
def write(n,x):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
 lock=open('/data/mlr_ifa/SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'STARTED.json').exists();write('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 jobs=[]
 for phase in ['train','evaluate']:
  for seed in (41,42,43):
   root=R/('seed%d'%seed)
   argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(root/'train.py'),'S'] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
   jobs.append((phase,seed,root,argv))
 for phase,seed,root,argv in jobs:
  name='train_S' if phase=='train' else 'evaluate'
  with (root/(name+'.log')).open('x') as f:
   p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
   write('STATUS.json',dict(phase=phase,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid()))
   rc=p.wait()
  (root/(name+'.exit')).write_text(str(rc));assert rc==0,(phase,seed,rc)
 write('STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));(R/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));(R/'controller.exit').write_text('1');raise
