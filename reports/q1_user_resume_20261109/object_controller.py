"""One finite three-seed run, maximum two trainers, serialized environments."""
import fcntl,hashlib,json,os,subprocess,time,traceback,uuid
from pathlib import Path
ENG=Path(__file__).resolve().parent;Q=ENG.parent.parent/'S_Q1_OBJECT41_42_43_20261109_rerun_v1';PY='/data/mlr_ifa/env/bin/python'
CASES=['Q1_41','Q1_42','Q1_43'];live={};completed=set()
def js(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def event(kind,**kw):
 with (ENG/'OBJECT_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n')
def verify():
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in js(ENG/'CONTRACT.json')['files'].items())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in js(Q/'CONTRACT.json')['files'].items())
def status(phase):write(ENG/'OBJECT_STATUS.json',dict(phase=phase,time=time.time(),controller=os.getpid(),active={k:v[0].pid for k,v in live.items()},completed=sorted(completed),max_training_concurrency=2))
def launch(case):
 assert case not in live and case not in completed and len(live)<2
 verify();r=Q/case;resuming=case in ['Q1_41','Q1_42'];exit_path=(ENG/(case+'_train.exit')) if resuming else r/'train_S.exit';log_path=(ENG/(case+'_train.log')) if resuming else r/'train_S.log';assert not exit_path.exists() and not log_path.exists()
 cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),str(ENG/'resume_wrapper.py' if resuming else Q/'train_wrapper.py'),case]
 log=log_path.open('x');p=subprocess.Popen(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 live[case]=(p,log);event('train_launch',case=case,pid=p.pid,command=cmd);status('train')
def finish(case,check=True):
 p,log=live[case];code=p.wait();log.close();del live[case]
 with ((ENG/(case+'_train.exit')) if case in ['Q1_41','Q1_42'] else Q/case/'train_S.exit').open('x') as f:f.write(str(code))
 event('train_exit',case=case,pid=p.pid,returncode=code)
 if not check:return
 assert code==0,(case,code)
 t=js(Q/case/'TRAIN_S.json');assert (t['epochs'],t['steps'],t['physical_optimizer_calls'],t['gate_blocks'],t['training_collection_episodes'])==(400,105500,195300,9100,40)
 assert all((Q/case/('collect_S_%d.exit'%s)).read_text().strip()=='0' for s in range(6,10))
 completed.add(case);status('train')
def main():
 assert not (Q/'controller.exit').exists() and not (ENG/'OBJECT_controller.exit').exists() and not (ENG/'OBJECT_EVENTS.jsonl').exists()
 verify();lock=(Q/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  pending=list(CASES)
  while pending or live:
   for c in list(live):
    if live[c][0].poll() is not None:finish(c)
   while pending and len(live)<2:launch(pending.pop(0))
   if live:time.sleep(3)
  assert completed==set(CASES)
  for case in CASES:
   r=Q/case;verify();assert not (r/'evaluate.exit').exists()
   with (r/'evaluate.log').open('x') as f:
    cmd=[PY,'-B',str(Q/'environment_locked.py'),case,'evaluate'];p=subprocess.Popen(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    event('evaluate_launch',case=case,pid=p.pid);status('evaluate_'+case);code=p.wait()
   with (r/'evaluate.exit').open('x') as f:f.write(str(code))
   event('evaluate_exit',case=case,pid=p.pid,returncode=code);assert code==0
   assert len(js(r/'OUTCOMES.json'))==200
  write(Q/'Q1_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,cases=CASES))
  verify();status('complete')
  with (ENG/'OBJECT_controller.exit').open('x') as f:f.write('0')
 except BaseException:
  write(ENG/'OBJECT_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),active={k:v[0].pid for k,v in live.items()}))
  for c in list(live):finish(c,check=False)
  with (ENG/'OBJECT_controller.exit').open('x') as f:f.write('1')
  raise
 finally:lock.close()
if __name__=='__main__':main()
