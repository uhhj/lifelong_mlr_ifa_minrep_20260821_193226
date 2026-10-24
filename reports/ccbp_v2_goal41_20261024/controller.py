import os,subprocess,json,time,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
def write(n,x):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
 lock=open('/data/mlr_ifa/CCBP_V2_GOAL41_20261024.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'STARTED.json').exists();write('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 jobs=[('prepare_B',['prepare','B'])]+[(phase+'_'+arm,[phase,arm]) for phase in ['admit','train'] for arm in 'BCNV']
 for name,args in jobs:
  write('STATUS.json',dict(phase=name,time=time.time(),controller=os.getpid()))
  with (R/(name+'.log')).open('x') as f:
   p=subprocess.Popen([PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(R/'train.py'),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'));write('STATUS.json',dict(phase=name,pid=p.pid,time=time.time(),controller=os.getpid()));rc=p.wait()
  (R/(name+'.exit')).write_text(str(rc));assert rc==0,(name,rc)
 write('STATUS.json',dict(phase='evaluate',time=time.time(),controller=os.getpid()))
 with (R/'evaluate.log').open('x') as f:
  p=subprocess.Popen([PY,'-B',str(R/'evaluate.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT);write('STATUS.json',dict(phase='evaluate',pid=p.pid,time=time.time(),controller=os.getpid()));rc=p.wait()
 (R/'evaluate.exit').write_text(str(rc));assert rc==0
 write('STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));(R/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));(R/'controller.exit').write_text('1');raise
