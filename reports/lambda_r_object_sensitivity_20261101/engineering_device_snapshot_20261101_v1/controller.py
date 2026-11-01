"""UNDEPLOYED. After validated optimization, wait for active run's natural exit."""
import os,sys,json,time,signal,fcntl,subprocess,hashlib,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY='/data/mlr_ifa/env/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def verify():
 for c in [Q/'CONTRACT.json',E/'CONTRACT.json']:
  assert all(sha(p)==h for p,h in js(c)['files'].items())
def main():
 own=(E/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STARTED.json').exists();verify();g=js(E/'GUARD.json');v=js(E/'VALIDATION.json')
 assert v['status']=='PASS_PAIRED_FULL_EPOCH_PATH_BOTH_LAMBDAS' and v['optimizer_completed']==v['optimizer_started']==7828 and v['environments']==0
 old=proc(g['controller']['pid']);child=proc(g['child']['pid']);assert old['start']==g['controller']['start'] and old['state']=='S' and child['start']==g['child']['start']
 assert js(Q/'STATUS.json')==g['queue_status'] and not (Q/'L005_42/train_S.log').exists()
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid())))
 with (E/'watchdog.log').open('x') as f:
  w=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (E/'WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'WATCHDOG_READY.json').exists()
 os.kill(old['pid'],signal.SIGSTOP)
 time.sleep(.1);assert proc(old['pid'])['state']=='T' and js(Q/'STATUS.json')==g['queue_status']
 assert not (Q/'L005_42/train_S.log').exists()
 put(E/'ARMED.json',dict(time=time.time(),status='WAIT_CURRENT_L005_41_NATURAL_EXIT',old_controller=old,current_child=child,watchdog_pid=w.pid,training_signals=0))
 while True:
  ch=proc(child['pid']);assert ch['start']==child['start'] and proc(old['pid'])['state']=='T'
  put(E/'STATUS.json',dict(phase='wait_L005_41',time=time.time(),controller=os.getpid(),child=ch))
  if ch['state']=='Z':break
  time.sleep(5)
 assert ch['exit_code']==0
 verify();r=Q/'L005_41';hist=js(r/'EPOCHS_S.json');t=js(r/'TRAIN_S.json')
 assert len(hist)==400 and t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 assert hist[:len(g['epoch_prefix'])]==g['epoch_prefix'] and (r/'GATES.jsonl').read_text().startswith(g['gate_prefix'])
 for st in range(6,10):
  assert sum(x['stage']==st for x in hist)==100
  assert (r/('collect_S_%d.exit'%st)).read_text().strip()=='0' and js(r/('MEMORY_S_%d.json'%st))['training_attempts']==10
 assert not (Q/'L005_42/train_S.log').exists()
 put(E/'NATURAL_EXIT.json',dict(time=time.time(),kernel_exit=ch,train_sha256=sha(r/'TRAIN_S.json'),epochs_sha256=sha(r/'EPOCHS_S.json'),training_signals=0))
 put(E/'HANDOFF_COMMITTED.json',dict(time=time.time(),old=old,source='NATURAL_EXIT.json'))
 assert proc(old['pid'])['start']==old['start'] and proc(old['pid'])['state']=='T'
 os.kill(old['pid'],signal.SIGKILL)
 lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a')
 for attempt in range(100):
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if attempt==99:raise
   time.sleep(.1)
 assert not (Q/'controller.exit').exists()
 with (r/'train_S.exit').open('x') as f:f.write('0')
 put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),source='natural launcher exit zero',root_lock=True,original_controller_success=False))
 for group in ['L005','L020']:
  if group=='L020':assert (Q/'L005_EVALUATION_COMPLETE.json').exists()
  for phase in ['train','evaluate']:
   for seed in [41,42,43]:
    case=group+'_'+str(seed)
    if phase=='train' and case=='L005_41':continue
    root=Q/case;name='train_S' if phase=='train' else 'evaluate';verify()
    assert not (root/(name+'.exit')).exists()
    argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
    with (root/(name+'.log')).open('x') as f:
     p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
     state=dict(phase=phase,suite=group,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid(),engineering_contract=sha(E/'CONTRACT.json'))
     put(Q/'STATUS.json',state);put(E/'STATUS.json',state);rc=p.wait()
    (root/(name+'.exit')).write_text(str(rc));assert rc==0,(case,phase,rc)
  assert all(len(js(Q/(group+'_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43])
  put(Q/(group+'_EVALUATION_COMPLETE.json'),dict(status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU',episodes=600,time=time.time()))
 put(E/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));put(Q/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid(),completion_source=str(E/'controller.exit')))
 (E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));(E/'controller.exit').write_text('1');raise
