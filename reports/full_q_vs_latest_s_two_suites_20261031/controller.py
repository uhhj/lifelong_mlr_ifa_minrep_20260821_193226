"""Unique finite Q dispatch, finishing each suite before starting the next."""
import os,subprocess,json,time,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
def write(n,x):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
 lock=open('/data/mlr_ifa/FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'STARTED.json').exists();write('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 for suite in ['GOAL','OBJECT']:
  if suite=='OBJECT':assert (R/'GOAL_EVALUATION_COMPLETE.json').exists()
  for phase in ['train','evaluate']:
   for seed in [41,42,43]:
    case=suite+'_'+str(seed);root=R/case;name='train_Q' if phase=='train' else 'evaluate'
    argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(root/'train.py'),'Q'] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
    assert not (root/(name+'.exit')).exists()
    with (root/(name+'.log')).open('x') as f:
     p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
     write('STATUS.json',dict(phase=phase,suite=suite,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid()))
     rc=p.wait()
    (root/(name+'.exit')).write_text(str(rc));assert rc==0,(case,phase,rc)
  assert all(len(json.loads((R/(suite+'_'+str(seed))/'OUTCOMES.json').read_text()))==200 for seed in [41,42,43])
  write(suite+'_EVALUATION_COMPLETE.json',dict(status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU',episodes=600,time=time.time()))
 write('STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));(R/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));(R/'controller.exit').write_text('1');raise
