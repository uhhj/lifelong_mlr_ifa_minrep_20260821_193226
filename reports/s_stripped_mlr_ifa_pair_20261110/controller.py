"""One finite controller: two training slots; each suite evaluated before next."""
from runtime import *
import fcntl,traceback
def launch(case):
 d=Q/'cases'/case;assert not (d/'TRAIN_DISPATCH.json').exists() and not (d/'train_A.exit').exists()
 ident=uuid.uuid4().hex;cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+ident,str(d/'train.py'),'A']
 log=(d/'train.log').open('x');p=subprocess.Popen(cmd,cwd=d,env=env(case),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);log.close()
 write(d/'TRAIN_DISPATCH.json',dict(pid=p.pid,time=time.time(),rdzv_id=ident,cmd=cmd));event('train_dispatch',case=case,pid=p.pid);return p
def train_suite(suite):
 pending=[c for c in CASES if c.startswith(suite+'_')];active={};failed=[]
 while pending or active:
  while pending and len(active)<2 and not failed:
   verify();capacity();case=pending.pop(0);active[case]=launch(case)
  for case,p in list(active.items()):
   rc=p.poll()
   if rc is None:continue
   d=Q/'cases'/case
   with (d/'train_A.exit').open('x') as f:f.write(str(rc)+'\n')
   event('train_exit',case=case,pid=p.pid,returncode=rc);del active[case]
   if rc:failed.append(dict(case=case,returncode=rc))
   else:
    t=js(d/'TRAIN_A.json');assert t['epochs']==400 and t['training_collection_episodes']==0
    assert t['steps']==(105500 if suite=='OBJECT' else 76100)
  write(Q/'STATUS.json',dict(status='PARTIAL_TRAINING' if not failed else 'FAILED_DRAINING_EXISTING_ONLY',suite=suite,time=time.time(),active={c:p.pid for c,p in active.items()},pending=pending,failures=failed))
  if failed and not active:raise RuntimeError(('Training failed; no automatic rerun',failed))
  if pending or active:time.sleep(10)
def evaluate(case):
 d=Q/'cases'/case;assert (d/'train_A.exit').read_text().strip()=='0'
 assert not (d/'EVALUATION_STARTED.json').exists() and not (d/'EVAL_DISPATCH.json').exists()
 lockpath=Path('/data/mlr_ifa/locks/s_stripped_pair_environment.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
 with lockpath.open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  def lockevent(kind):
   with (Q/'ENVIRONMENT_LOCK_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,case=case,time=time.time(),owner=os.getpid()))+'\n');f.flush()
  lockevent('acquired');cmd=[PY,'-B',str(d/'evaluate.py')]
  with (d/'evaluate.log').open('x') as log:p=subprocess.Popen(cmd,cwd=d,env=env(case),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(d/'EVAL_DISPATCH.json',dict(pid=p.pid,time=time.time(),cmd=cmd));event('evaluate_dispatch',case=case,pid=p.pid)
  write(Q/'STATUS.json',dict(status='PARTIAL_EVALUATING',case=case,pid=p.pid,time=time.time()))
  rc=p.wait()
  with (d/'evaluate.exit').open('x') as f:f.write(str(rc)+'\n')
  lockevent('released');event('evaluate_exit',case=case,pid=p.pid,returncode=rc)
  assert rc==0,('Formal evaluation failed; no retry',case,rc)
  assert len(js(d/'OUTCOMES.json'))==200 and js(d/'EVALUATION_COMPLETE.json')['episodes']==200
def main():
 with (Q/'controller.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (Q/'controller.exit').exists();verify();capacity();event('controller_start',pid=os.getpid())
  for suite in ['OBJECT','GOAL']:
   train_suite(suite)
   for case in [c for c in CASES if c.startswith(suite+'_')]:verify();capacity();evaluate(case)
   write(Q/(suite+'_COMPLETE.json'),dict(status='PASS_EXECUTION_PENDING_CPU_ACCEPTANCE',suite=suite,training_runs=6,formal=1200,time=time.time()))
  verify();event('controller_complete',pid=os.getpid());write(Q/'STATUS.json',dict(status='EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE',time=time.time()))
if __name__=='__main__':
 rc=1
 try:main();rc=0
 except BaseException:write(Q/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
 finally:
  with (Q/'controller.exit').open('x') as f:f.write(str(rc)+'\n')
