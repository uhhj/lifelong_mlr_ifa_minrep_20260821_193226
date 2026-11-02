"""Unique restart of verified partial run, with independent rendezvous for every launcher."""
import os,sys,json,time,signal,fcntl,subprocess,hashlib,traceback,uuid
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY='/data/mlr_ifa/env/bin/python';live={}
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def verify():
 for p in [Q/'CONTRACT.json',*Q.glob('engineering_*/CONTRACT.json')]:assert all(sha(n)==h for n,h in js(p)['files'].items())
def state(phase):
 x=dict(time=time.time(),controller=os.getpid(),phase=phase,active={k:dict(pid=p.pid,phase=ph) for k,(p,ph) in live.items()},engineering_contract=sha(E/'CONTRACT.json'),completion_source=str(E/'controller.exit'),max_training_concurrency=2)
 put(E/'STATUS.json',x);put(Q/'STATUS.json',x)
def paths(case,phase):
 if case=='L005_43' and phase=='train':return E/'L005_43_resume.log',E/'L005_43_resume.exit'
 name='train_S' if phase=='train' else 'evaluate';return Q/case/(name+'.log'),Q/case/(name+'.exit')
def launch(case,phase):
 verify();log,exit=paths(case,phase);assert not log.exists() and not exit.exists();rid=str(uuid.uuid4())
 argv=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+rid,str(E/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(Q/case/'evaluate.py')]
 with log.open('x') as f:p=subprocess.Popen(argv,cwd=Q/case,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 live[case]=(p,phase)
 with (E/'LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),case=case,phase=phase,pid=p.pid,argv=argv,rdzv_id=rid if phase=='train' else None))+'\n')
 state(phase)
def finish(case,block=False):
 p,phase=live[case];rc=p.wait() if block else p.poll()
 if rc is None:return None
 _,out=paths(case,phase)
 with out.open('x') as f:f.write(str(rc))
 put(E/(case+'_'+phase+'_EXIT.json'),dict(pid=p.pid,returncode=rc,time=time.time()));del live[case];return rc
def one(case,phase):launch(case,phase);assert finish(case,True)==0,(case,phase)
def train_pool(cases):
 pending=list(cases);failure=None
 while pending or live:
  while pending and len(live)<2 and failure is None:launch(pending.pop(0),'train')
  for c in list(live):
   rc=finish(c)
   if rc is not None and rc!=0:failure=(c,rc)
  state('train_parallel')
  if failure:
   for c in list(live):finish(c,True)
   raise RuntimeError(str(failure))
  if live:time.sleep(2)
def main():
 lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (E/'STARTED.json').exists();verify()
 v=js(E/'VALIDATION.json');assert v['status']=='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK' and v['started']==v['completed']==869
 assert js(E/'LAUNCHER_VALIDATION.json')['early_peer_exit_survived']
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),root_lock=True,production_signals=0))
 one('L005_43','train');t=js(Q/'L005_43/TRAIN_S.json');assert t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 for seed in [41,42,43]:one('L005_'+str(seed),'evaluate')
 assert all(len(js(Q/('L005_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L005_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 train_pool(['L020_41','L020_42','L020_43'])
 for seed in [41,42,43]:one('L020_'+str(seed),'evaluate')
 assert all(len(js(Q/('L020_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L020_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 state('complete');(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),production_signals=0))
  for c in list(live):
   try:finish(c,True)
   except BaseException:pass
  (E/'controller.exit').write_text('1');raise
