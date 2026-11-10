"""Continue only unfinished evaluations and GOAL; never repeat OBJECT training."""
from resume_runtime import *
import fcntl,uuid,traceback
def launch(case,phase):
 original.capacity();source=Q/'cases'/case/'source'
 assert not (E/(case+'_'+phase+'.exit')).exists()
 if phase=='train':
  assert case.startswith('GOAL_') and not (P/case/'train.log').exists()
  cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),'--module','coreset_exp.train','random']
 else:cmd=[PY,'-B',str(E/'evaluate_guarded.py'),case]
 with (E/(case+'_'+phase+'.log')).open('x') as f:p=subprocess.Popen(cmd,cwd=source,env=env(case),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 event('launch',case=case,phase=phase,pid=p.pid,command=cmd);return p
def reap(case,phase,p):
 code=p.wait();(E/(case+'_'+phase+'.exit')).write_text(str(code)+'\n')
 if phase=='train':
  with (P/case/'train.exit').open('x') as f:f.write(str(code)+'\n')
 event('exit',case=case,phase=phase,pid=p.pid,code=code);return code
def evaluate_suite(suite):
 for seed in [41,42,43]:
  c=suite+'_'+str(seed)
  assert js(P/c/'random/TRAINING_COMPLETE.json')['epochs']==400
  write(E/'STATUS.json',dict(status='EVALUATING',case=c,observed=time.time()))
  p=launch(c,'evaluate');assert reap(c,'evaluate',p)==0,'Evaluation failed; preserve attempt, no retry'
  assert js(P/c/'random/evaluation/EVALUATION_COMPLETE.json')['episodes']==200
 write(E/(suite+'_COMPLETE.json'),dict(epochs=1200,episodes=600,observed=time.time()))
def main():
 lock=(Q/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'EVENTS.jsonl').exists() and not (E/'controller.exit').exists()
 active={}
 try:
  verify();evaluate_suite('OBJECT')
  pending=['GOAL_'+str(s) for s in [41,42,43]];failed=False
  while pending or active:
   while pending and len(active)<2 and not failed:
    c=pending.pop(0);active[c]=launch(c,'train')
   for c,p in list(active.items()):
    if p.poll() is not None:
     if reap(c,'train',p)!=0:failed=True
     del active[c]
   write(E/'STATUS.json',dict(status='TRAINING' if not failed else 'FAILURE_WAITING_EXISTING_RUNS',suite='GOAL',active={c:p.pid for c,p in active.items()},pending=pending,observed=time.time()))
   if failed and not active:raise RuntimeError('Training failure; no retry')
   if active:time.sleep(10)
  evaluate_suite('GOAL');verify();assert not processes([P,Q/'cases'])
  write(E/'STATUS.json',dict(status='NATURAL_COMPLETE_PENDING_CPU_ACCEPTANCE',observed=time.time()))
  (E/'controller.exit').write_text('0\n')
 except BaseException:
  write(E/'FAILURE.json',dict(error=traceback.format_exc(),observed=time.time()))
  for c,p in list(active.items()):reap(c,'train',p)
  (E/'controller.exit').write_text('1\n');raise
 finally:lock.close()
if __name__=='__main__':main()
