"""Single-use scheduling handoff; no signals to training or evaluation children."""
import os,json,time,signal,fcntl,subprocess,traceback,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY='/data/mlr_ifa/env/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19],exit_code=int(s[49]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def verify():
 for c in [Q/'CONTRACT.json',E/'CONTRACT.json']:
  assert all(sha(p)==h for p,h in js(c)['files'].items())
def completed(case):
 rows=js(Q/case/'EPOCHS_C.json');t=js(Q/case/'TRAIN_C.json');budget=142200 if case.startswith('GOAL') else 195300
 assert len(rows)==t['epochs']==700 and t['steps']==t['physical_optimizer_calls']==budget and t['isolated_optimizer_calls']==91
 assert [sum(x['stage']==s for x in rows) for s in [6,7,8,9]]==[100,200,200,200]
 return dict(epochs_sha256=sha(Q/case/'EPOCHS_C.json'),train_sha256=sha(Q/case/'TRAIN_C.json'))
def main():
 ownlock=(E/'controller.lock').open('a');fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STARTED.json').exists(),'Never repeat scheduling dispatch'
 verify();g=js(E/'GUARD.json');old=proc(g['controller']['pid']);child=proc(g['child']['pid'])
 assert old['start']==g['controller']['start'] and child['start']==g['child']['start'] and child['ppid']==old['pid']
 assert old['state'] not in ['T','Z'] and child['state']!='Z'
 assert js(Q/'STATUS.json')['case']=='GOAL_42' and not (Q/'GOAL_43/train_C.log').exists()
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid()),contract_sha256=sha(E/'CONTRACT.json')))
 with (E/'watchdog.log').open('x') as f:
  watch=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (E/'WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'WATCHDOG_READY.json').exists()
 # PID-specific STOP only affects the waiting coordinator, never its child group.
 assert proc(old['pid'])['start']==old['start']
 os.kill(old['pid'],signal.SIGSTOP)
 for _ in range(100):
  if proc(old['pid'])['state']=='T':break
  time.sleep(.05)
 assert proc(old['pid'])['state']=='T'
 assert js(Q/'STATUS.json')['case']=='GOAL_42' and not (Q/'GOAL_43/train_C.log').exists()
 state=dict(phase='wait_GOAL_42_natural_exit',time=time.time(),controller=os.getpid(),original_controller=old,child=child,watchdog_pid=watch.pid,training_signals=0)
 put(E/'ARMED.json',state);put(E/'STATUS.json',state)
 while True:
  child=proc(g['child']['pid']);assert child['start']==g['child']['start']
  oldnow=proc(old['pid']);assert oldnow['start']==old['start'] and oldnow['state']=='T'
  if child['state']=='Z':break
  time.sleep(5)
 assert child['exit_code']==0,child
 verify();done=completed('GOAL_42')
 assert all(sha(p)==h for p,h in g['completed_GOAL41'].items())
 assert js(Q/'GOAL_42/EPOCHS_C.json')[:len(g['GOAL42_prefix'])]==g['GOAL42_prefix']
 assert not (Q/'GOAL_43/train_C.log').exists() and not (Q/'controller.exit').exists()
 receipt=dict(status='NATURAL_GOAL42_EXIT_VERIFIED',time=time.time(),kernel_exit=child,training_signals=0,completed=done)
 put(E/'GOAL42_EXIT_OBSERVATION.json',receipt)
 put(E/'HANDOFF_COMMITTED.json',dict(time=time.time(),old=old,new_controller=os.getpid(),exit_receipt_sha256=sha(E/'GOAL42_EXIT_OBSERVATION.json')))
 # The only killed process is the paused coordinator, after its child is a successful zombie.
 assert proc(old['pid'])['start']==old['start'] and proc(old['pid'])['state']=='T'
 os.kill(old['pid'],signal.SIGKILL)
 rootlock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a')
 for i in range(100):
  try:fcntl.flock(rootlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if i==99:raise
   time.sleep(.1)
 with (Q/'GOAL_42/train_C.exit').open('x') as f:f.write('0')
 put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),root_lock_acquired=True,original_controller_success_marker=False,GOAL42_exit_source='kernel exit status, GOAL42_EXIT_OBSERVATION.json'))
 jobs=[('train','GOAL_43')]+[('evaluate','GOAL_'+str(s)) for s in [41,42,43]]+[('train','OBJECT_'+str(s)) for s in [41,42,43]]+[('evaluate','OBJECT_'+str(s)) for s in [41,42,43]]
 put(E/'QUEUE.json',dict(jobs=jobs,science_unchanged=True,new_S_runs=0))
 for phase,case in jobs:
  verify();root=Q/case;name='train_C' if phase=='train' else 'evaluate'
  assert not (root/(name+'.exit')).exists()
  if phase=='evaluate':
   assert (root/'train_C.exit').read_text()=='0';completed(case)
  argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(root/'train.py'),'C'] if phase=='train' else [PY,'-B',str(root/'evaluate.py')]
  with (root/(name+'.log')).open('x') as f:
   p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
   state=dict(phase=phase,case=case,pid=p.pid,time=time.time(),controller=os.getpid(),engineering_contract=sha(E/'CONTRACT.json'))
   put(Q/'STATUS.json',state);put(E/'STATUS.json',state);rc=p.wait()
  with (root/(name+'.exit')).open('x') as f:f.write(str(rc))
  assert rc==0,(phase,case,rc)
  if phase=='evaluate' and case=='GOAL_43':
   for s in [41,42,43]:assert len(js(Q/('GOAL_'+str(s))/'OUTCOMES.json'))==200
   put(E/'GOAL_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,next_phase='OBJECT_training'))
 put(E/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()))
 put(Q/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid(),completion_source=str(E/'controller.exit')))
 (E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));(E/'controller.exit').write_text('1');raise
