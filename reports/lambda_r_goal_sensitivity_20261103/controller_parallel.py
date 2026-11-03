"""Single-use GOAL scheduler; at most two trainers, all environment work serialized."""
import fcntl,hashlib,json,os,subprocess,time,traceback,uuid
from pathlib import Path
Q=Path(__file__).resolve().parent
PY='/data/mlr_ifa/env/bin/python'
CASES=[g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]
live={};completed=set()
def js(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def event(kind,**kw):
 with (Q/'CONTROLLER_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n')
def verify():
 c=js(Q/'CONTRACT.json')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
def status(phase):
 write(Q/'CONTROLLER_STATUS.json',dict(phase=phase,time=time.time(),controller=os.getpid(),active={k:v[0].pid for k,v in live.items()},completed=sorted(completed),max_training_concurrency=2))
def launch(case):
 assert case not in live and case not in completed and len(live)<2
 verify();r=Q/case;assert not (r/'train_S.exit').exists() and not (r/'train_S.log').exists()
 cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),str(Q/'train_wrapper.py'),case]
 log=(r/'train_S.log').open('x');p=subprocess.Popen(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 live[case]=(p,log);event('train_launch',case=case,pid=p.pid,command=cmd);status('train')
def finish(case,check=True):
 p,log=live[case];code=p.wait();log.close();del live[case]
 with (Q/case/'train_S.exit').open('x') as f:f.write(str(code))
 event('train_exit',case=case,pid=p.pid,returncode=code)
 if not check:return
 assert code==0,(case,code)
 t=js(Q/case/'TRAIN_S.json');assert (t['epochs'],t['steps'],t['physical_optimizer_calls'],t['gate_blocks'],t['training_collection_episodes'])==(400,76100,142200,6700,40)
 assert all((Q/case/('collect_S_%d.exit'%s)).read_text().strip()=='0' for s in range(6,10))
 completed.add(case);status('train')
def poll():
 for c in list(live):
  if live[c][0].poll() is not None:finish(c)
def evaluate(group):
 assert all(group+'_'+str(s) in completed for s in (41,42,43))
 for seed in (41,42,43):
  case=group+'_'+str(seed);r=Q/case;verify()
  assert not (r/'evaluate.exit').exists()
  with (r/'evaluate.log').open('x') as f:
   cmd=[PY,'-B',str(Q/'environment_locked.py'),case,'evaluate'];p=subprocess.Popen(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   event('evaluate_launch',case=case,pid=p.pid);status('evaluate_'+case);code=p.wait()
  with (r/'evaluate.exit').open('x') as f:f.write(str(code))
  event('evaluate_exit',case=case,pid=p.pid,returncode=code);assert code==0
  assert len(js(r/'OUTCOMES.json'))==200
  poll()
 write(Q/(group+'_EVALUATION_COMPLETE.json'),dict(time=time.time(),episodes=600,cases=[group+'_'+str(s) for s in (41,42,43)]))
def main():
 assert not (Q/'controller.exit').exists() and not (Q/'CONTROLLER_EVENTS.jsonl').exists()
 verify()
 lock=(Q/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  pending=['L005_41','L005_42','L005_43','L020_41']
  while not all('L005_'+str(s) in completed for s in (41,42,43)):
   poll()
   while pending and len(live)<2:launch(pending.pop(0))
   time.sleep(3)
  evaluate('L005')
  pending=pending+['L020_42','L020_43']
  while pending or live:
   poll()
   while pending and len(live)<2:launch(pending.pop(0))
   if live:time.sleep(3)
  evaluate('L020');verify();status('complete')
  with (Q/'controller.exit').open('x') as f:f.write('0')
 except BaseException:
  write(Q/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),active={k:v[0].pid for k,v in live.items()}))
  for c in list(live):finish(c,check=False)
  with (Q/'controller.exit').open('x') as f:f.write('1')
  raise
 finally:lock.close()
if __name__=='__main__':main()
