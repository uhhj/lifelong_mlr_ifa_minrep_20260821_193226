"""UNDEPLOYED: natural current-run boundary, then at most two L020 training jobs."""
import os,sys,json,time,signal,fcntl,subprocess,hashlib,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY='/data/mlr_ifa/env/bin/python'
live={}
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]))
def verify():
 for name in ['CONTRACT.json','engineering_device_snapshot_20261101_v1/CONTRACT.json','engineering_resume_3090_20261102_v1/CONTRACT.json','engineering_checks_3090_20261102_v1/CONTRACT.json',E.name+'/CONTRACT.json']:
  assert all(sha(p)==h for p,h in js(Q/name)['files'].items())
def state(phase):
 x=dict(time=time.time(),controller=os.getpid(),phase=phase,active={k:dict(pid=v[0].pid,phase=v[1]) for k,v in live.items()},engineering_contract=sha(E/'CONTRACT.json'),completion_source=str(E/'controller.exit'),max_training_concurrency=2)
 put(E/'STATUS.json',x);put(Q/'STATUS.json',x)
def launch(case,phase):
 verify();root=Q/case;name='train_S' if phase=='train' else 'evaluate';assert not (root/(name+'.exit')).exists()
 argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
 with (root/(name+'.log')).open('x') as f:p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
 live[case]=(p,phase);state('train_parallel' if len(live)>1 else phase)
def finish(case,block=False):
 p,phase=live[case];rc=p.wait() if block else p.poll()
 if rc is None:return None
 name='train_S' if phase=='train' else 'evaluate';put(E/(case+'_'+phase+'_EXIT.json'),dict(pid=p.pid,returncode=rc,time=time.time(),own_child_wait=True))
 with (Q/case/(name+'.exit')).open('x') as f:f.write(str(rc))
 del live[case];return rc
def one(case,phase):
 launch(case,phase);rc=finish(case,True);assert rc==0,(case,phase,rc)
def train_pool(cases):
 pending=list(cases);failure=None
 while pending or live:
  while pending and len(live)<2 and failure is None:launch(pending.pop(0),'train')
  for case in list(live):
   rc=finish(case)
   if rc is not None and rc!=0:failure=(case,rc)
  state('train_parallel')
  if failure is not None:
   # Let already-started peers finish without a signal; never launch replacements.
   for case in list(live):finish(case,True)
   raise RuntimeError(str(failure))
  if live:time.sleep(2)
def prefix_guard(g):
 r=Q/'L005_42';h=js(r/'EPOCHS_S.json');raw=json.dumps(h[:g['epoch_count']],sort_keys=True,separators=(',',':')).encode()
 assert hashlib.sha256(raw).hexdigest()==g['epoch_prefix_sha256']
 with (r/'GATES.jsonl').open('rb') as f:assert hashlib.sha256(f.read(g['gate_prefix_bytes'])).hexdigest()==g['gate_prefix_sha256']
 assert all(sha(p)==h for p,h in g['completed_L005_41_hashes'].items())
def main():
 own=(E/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (E/'STARTED.json').exists();verify();g=js(E/'GUARD.json')
 assert js(E/'PARALLEL_ADMISSION.json')['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
 old=proc(g['controller']['pid']);child=proc(g['child']['pid']);assert old['start']==g['controller']['start'] and old['state']=='S' and child['start']==g['child']['start']
 assert js(Q/'STATUS.json')==g['queue_status'] and not (Q/'L005_43/train_S.log').exists()
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid())))
 with (E/'watchdog.log').open('x') as f:w=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (E/'WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'WATCHDOG_READY.json').exists();os.kill(old['pid'],signal.SIGSTOP);time.sleep(.1)
 assert proc(old['pid'])['state']=='T' and js(Q/'STATUS.json')==g['queue_status'];prefix_guard(g)
 put(E/'ARMED.json',dict(time=time.time(),status='WAIT_L005_42_NATURAL_EXIT',watchdog=w.pid,training_signals=0))
 while True:
  ch=proc(child['pid']);assert ch['start']==child['start'] and proc(old['pid'])['state']=='T'
  put(E/'STATUS.json',dict(phase='wait_L005_42',time=time.time(),controller=os.getpid(),child=ch))
  if ch['state']=='Z':break
  time.sleep(5)
 assert ch['exit_code']==0;verify();prefix_guard(g);r=Q/'L005_42';t=js(r/'TRAIN_S.json')
 assert t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 assert len(js(r/'EPOCHS_S.json'))==400
 for st in range(6,10):assert (r/('collect_S_%d.exit'%st)).read_text().strip()=='0' and js(r/('MEMORY_S_%d.json'%st))['training_attempts']==10
 assert not (Q/'L005_43/train_S.log').exists();put(E/'NATURAL_EXIT.json',dict(time=time.time(),kernel_exit=ch,train_sha256=sha(r/'TRAIN_S.json'),training_signals=0))
 put(E/'HANDOFF_COMMITTED.json',dict(time=time.time(),old=old));assert proc(old['pid'])['start']==old['start'] and proc(old['pid'])['state']=='T';os.kill(old['pid'],signal.SIGKILL)
 lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a')
 for attempt in range(100):
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if attempt==99:raise
   time.sleep(.1)
 for name in ['controller.exit','engineering_device_snapshot_20261101_v1/controller.exit','engineering_resume_3090_20261102_v1/controller.exit','engineering_checks_3090_20261102_v1/controller.exit']:assert not (Q/name).exists()
 with (r/'train_S.exit').open('x') as f:f.write('0')
 put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),source='natural launcher exit zero',root_lock=True,original_controller_success=False))
 one('L005_43','train')
 for seed in [41,42,43]:one('L005_'+str(seed),'evaluate')
 assert all(len(js(Q/('L005_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L005_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 train_pool(['L020_41','L020_42','L020_43'])
 for seed in [41,42,43]:one('L020_'+str(seed),'evaluate')
 assert all(len(js(Q/('L020_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L020_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 state('complete');(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc()
  for case in list(live):
   try:finish(case,True)
   except BaseException:pass
  put(E/'FAILURE.json',dict(time=time.time(),error=error,training_signals=0));(E/'controller.exit').write_text('1');raise
