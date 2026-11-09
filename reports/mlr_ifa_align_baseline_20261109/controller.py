"""OBJECT three training runs then evaluation; GOAL same. Max two trainers."""
from runtime import *
import fcntl,uuid,traceback
def launch(case,kind):
 capacity();d=P/case;source=Q/'cases'/case/'source'
 assert not (d/(kind+'.exit')).exists() and not (d/(kind+'.log')).exists()
 if kind=='train':
  cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),'--module','coreset_exp.train','random']
 else:cmd=[PY,'-B',str(Q/'evaluate_locked.py'),case]
 with (d/(kind+'.log')).open('x') as f:p=subprocess.Popen(cmd,cwd=source,env=env(case),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 event('launch',case=case,phase=kind,pid=p.pid,command=cmd);return p
def reap(case,kind,p):
 code=p.wait();(P/case/(kind+'.exit')).write_text(str(code)+'\n');event('exit',case=case,phase=kind,pid=p.pid,code=code)
 return code
def main():
 lock=(Q/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (Q/'EVENTS.jsonl').exists() and not (Q/'controller.exit').exists()
 active={}
 try:
  verify();acceptance_gate();capacity()
  for suite in ['OBJECT','GOAL']:
   pending=[suite+'_'+str(s) for s in [41,42,43]];failed=False
   while pending or active:
    while pending and len(active)<2 and not failed:
     c=pending.pop(0);active[c]=launch(c,'train')
    for c,p in list(active.items()):
     if p.poll() is not None:
      if reap(c,'train',p)!=0:failed=True
      del active[c]
    write(Q/'STATUS.json',dict(status='TRAINING' if not failed else 'FAILURE_WAITING_EXISTING_RUNS_NO_NEW_LAUNCH',suite=suite,active={c:p.pid for c,p in active.items()},pending=pending,observed=time.time()))
    if failed and not active:raise RuntimeError('Training failed; originals preserved, no retry')
    if active:time.sleep(10)
   for c in [suite+'_'+str(s) for s in [41,42,43]]:
    assert js(P/c/'random/TRAINING_COMPLETE.json')['epochs']==400
    p=launch(c,'evaluate');assert reap(c,'evaluate',p)==0,'Evaluation failed; no retry'
    assert js(P/c/'random/evaluation/EVALUATION_COMPLETE.json')['episodes']==200
   write(Q/(suite+'_COMPLETE.json'),dict(epochs=1200,episodes=600,observed=time.time()))
  verify();assert not processes([P,Q/'cases'])
  write(Q/'STATUS.json',dict(status='NATURAL_COMPLETE_PENDING_CPU_ACCEPTANCE',epochs=2400,episodes=1200,observed=time.time()))
  (Q/'controller.exit').write_text('0\n')
 except BaseException:
  write(Q/'FAILURE.json',dict(error=traceback.format_exc(),observed=time.time(),no_retry=True))
  # Existing trainers finish naturally even if a later launch/precheck fails.
  for c,p in list(active.items()):reap(c,'train',p)
  (Q/'controller.exit').write_text('1\n');raise
 finally:lock.close()
if __name__=='__main__':main()
