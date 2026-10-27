"""Finite 5 training runs then 5 evaluations. Single dispatch, no retries."""
import os,subprocess,json,time,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
CASES=['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']
def write(n,x):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
 lock=open('/data/mlr_ifa/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'STARTED.json').exists();write('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 for phase in ['train','evaluate']:
  for case in CASES:
   root=R/case;name='train_C' if phase=='train' else 'evaluate'
   argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(root/'train.py'),'C'] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
   with (root/(name+'.log')).open('x') as f:
    p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
    write('STATUS.json',dict(phase=phase,case=case,pid=p.pid,time=time.time(),controller=os.getpid()));rc=p.wait()
   (root/(name+'.exit')).write_text(str(rc));assert rc==0,(phase,case,rc)
 write('STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));(R/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));(R/'controller.exit').write_text('1');raise
