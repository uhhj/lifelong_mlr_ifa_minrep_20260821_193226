"""Single-use boundary handoff: wait42, train43 optimized, unchanged eval41/42/43."""
import os,sys,time,json,hashlib,signal,fcntl,subprocess,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY='/data/mlr_ifa/env/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def verify():
 for name in [Q/'CONTRACT.json',E/'CONTRACT.json']:
  for p,h in js(name)['files'].items():assert sha(p)==h,p
def main():
 ownlock=(E/'controller.lock').open('a');fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STARTED.json').exists(),'Never rerun this controller'
 verify();g=js(E/'GUARD.json');validation=js(E/'VALIDATION.json')
 assert validation['status']=='PASS_FULL_TRAINING_PATH_EXACT' and validation['optimizer_calls_completed']==3914 and validation['environments']==0
 assert not (Q/'seed43/train_S.log').exists()
 old=proc(3013);assert old['start']==g['controller']['start'] and old['state']=='T'
 started=dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid()),validation_sha256=sha(E/'VALIDATION.json'),contract_sha256=sha(E/'CONTRACT.json'))
 put(E/'STARTED.json',started)
 # Replace bounded validation watchdog with a live-owner watchdog. Only the waiting
 # original coordinator is resumed if this new owner dies BEFORE a committed handoff.
 with (E/'owner_watchdog.log').open('x') as f:
  watch=subprocess.Popen([PY,'-B',str(E/'owner_watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(50):
  if (E/'OWNER_WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'OWNER_WATCHDOG_READY.json').exists()
 wp=proc(g['watchdog_pid']);assert wp['cmd'].strip().endswith(str(E/'watchdog.py'))
 os.kill(g['watchdog_pid'],signal.SIGTERM)
 put(E/'ARMED.json',dict(status='WAITING_FOR_SEED42_NATURAL_EXIT',time=time.time(),seed42=g['child'],owner_watchdog=watch.pid))
 while True:
  child=proc(g['child']['pid']);assert child['start']==g['child']['start']
  assert proc(3013)['state']=='T'
  put(E/'STATUS.json',dict(phase='wait_seed42',time=time.time(),pid=os.getpid(),child=child))
  if child['state']=='Z':break
  time.sleep(5)
 assert child['exit_code']==0,child
 verify();r=Q/'seed42';h=js(r/'EPOCHS_S.json');t=js(r/'TRAIN_S.json')
 assert len(h)==400 and t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 assert [sum(x['stage']==s for x in h) for s in [6,7,8,9]]==[100]*4
 for stage in [6,7,8,9]:
  assert (r/('collect_S_%d.exit'%stage)).read_text().strip()=='0'
  m=js(r/('MEMORY_S_%d.json'%stage));assert m['training_attempts']==10
 assert not (Q/'seed43/train_S.log').exists()
 # The launcher is now a zombie with kernel-reported exit status zero; training and
 # collection are finished. No signal has been sent to either seed42 process.
 receipt=dict(status='NATURAL_SEED42_EXIT_VERIFIED',time=time.time(),kernel_exit=child,train_report_sha256=sha(r/'TRAIN_S.json'),epochs_sha256=sha(r/'EPOCHS_S.json'),validation_sha256=sha(E/'VALIDATION.json'),training_signals=0)
 put(E/'SEED42_EXIT_OBSERVATION.json',receipt)
 put(E/'HANDOFF_COMMITTED.json',dict(time=time.time(),old_coordinator=old,new_coordinator=started,source=receipt))
 # Kill only the already-paused coordinator after its child has exited naturally.
 assert proc(3013)['start']==old['start'] and proc(3013)['state']=='T'
 os.kill(3013,signal.SIGKILL)
 rootlock=Path('/data/mlr_ifa/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1.lock').open('a')
 for attempt in range(100):
  try:fcntl.flock(rootlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if attempt==99:raise
   time.sleep(.1)
 assert not (Q/'controller.exit').exists(),'Do not invent original coordinator success'
 with (r/'train_S.exit').open('x') as f:f.write('0')
 put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),seed42_exit_source='SEED42_EXIT_OBSERVATION.json: kernel wait status',original_controller_success_marker=False,root_lock_acquired=True))
 jobs=[('train',43,[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'train43_optimized.py'),'S'])]
 jobs += [('evaluate',seed,[PY,'-B',str(Q/('seed%d'%seed)/'evaluate.py')]) for seed in [41,42,43]]
 for phase,seed,argv in jobs:
  verify();root=Q/('seed%d'%seed);name='train_S' if phase=='train' else 'evaluate'
  assert not (root/(name+'.exit')).exists()
  with (root/(name+'.log')).open('x') as f:
   p=subprocess.Popen(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
   state=dict(phase=phase,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid(),engineering_contract=sha(E/'CONTRACT.json'))
   put(Q/'STATUS.json',state);put(E/'STATUS.json',state);rc=p.wait()
  (root/(name+'.exit')).write_text(str(rc));assert rc==0,(phase,seed,rc)
 put(E/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));put(Q/'STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid(),completion_source=str(E/'controller.exit')))
 (E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));(E/'controller.exit').write_text('1');raise
