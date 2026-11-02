"""User-authorized cross-lambda overlap, unchanged active seed43 and independent rendezvous."""
import os,sys,json,time,signal,fcntl,subprocess,hashlib,traceback,uuid
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PREV=Q/'engineering_rendezvous_recovery_20261102_v1';PY='/data/mlr_ifa/env/bin/python';live={};adopted=None
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def proc(pid):
 s=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]))
def same(i):
 p=proc(i['pid']);assert p['start']==i['start'];return p
def verify():
 for p in [Q/'CONTRACT.json',*Q.glob('engineering_*/CONTRACT.json')]:assert all(sha(n)==h for n,h in js(p)['files'].items())
def state(phase):
 active={k:dict(pid=p.pid,phase=ph) for k,(p,ph) in live.items()}
 if adopted:active['L005_43']=dict(pid=adopted['pid'],phase='train',adopted_existing_process=True)
 assert sum(x['phase']=='train' for x in active.values())<=2
 put(Q/'STATUS.json',dict(time=time.time(),controller=os.getpid(),phase=phase,active=active,engineering_contract=sha(E/'CONTRACT.json'),completion_source=str(E/'controller.exit'),max_training_concurrency=2))
 put(E/'STATUS.json',js(Q/'STATUS.json'))
def launch(case,phase):
 verify();name='train_S' if phase=='train' else 'evaluate';log=Q/case/(name+'.log');exit=Q/case/(name+'.exit');assert not log.exists() and not exit.exists();rid=str(uuid.uuid4())
 argv=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+rid,str(PREV/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(E/'evaluate_locked.py'),case]
 with log.open('x') as f:p=subprocess.Popen(argv,cwd=Q/case,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 live[case]=(p,phase)
 with (E/'LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),case=case,phase=phase,pid=p.pid,argv=argv,rdzv_id=rid if phase=='train' else None))+'\n')
 state(phase)
def finish(case,block=False):
 p,phase=live[case];rc=p.wait() if block else p.poll()
 if rc is None:return None
 name='train_S' if phase=='train' else 'evaluate'
 with (Q/case/(name+'.exit')).open('x') as f:f.write(str(rc))
 put(E/(case+'_'+phase+'_EXIT.json'),dict(pid=p.pid,returncode=rc,time=time.time()));del live[case];return rc
def complete(case):
 t=js(Q/case/'TRAIN_S.json');assert t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 for stage in range(6,10):assert (Q/case/('collect_S_%d.exit'%stage)).read_text().strip()=='0'
def prefix(g):
 h=js(Q/'L005_43/EPOCHS_S.json');assert hashlib.sha256(json.dumps(h[:g['epoch_count']],sort_keys=True,separators=(',',':')).encode()).hexdigest()==g['epoch_prefix_sha256']
 with (Q/'L005_43/GATES.jsonl').open('rb') as f:assert hashlib.sha256(f.read(g['gate_prefix_bytes'])).hexdigest()==g['gate_prefix_sha256']
 assert all(sha(p)==h for p,h in g['completed_hashes'].items())
def train_pool(cases):
 pending=list(cases);failure=None
 while pending or live:
  while pending and len(live)<2 and failure is None:launch(pending.pop(0),'train')
  for case in list(live):
   rc=finish(case)
   if rc is not None:
    if rc==0:complete(case)
    else:failure=(case,rc)
  state('L020_train_parallel')
  if failure:
   for case in list(live):finish(case,True)
   raise RuntimeError(str(failure))
  if live:time.sleep(2)
def main():
 global adopted
 own=(E/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (E/'STARTED.json').exists();verify();g=js(E/'GUARD.json')
 assert js(PREV/'LAUNCHER_VALIDATION.json')['early_peer_exit_survived']
 assert same(g['old_controller'])['state'] in ['S','R','D'];assert same(g['child'])['state']!='Z' and same(g['training'])['state'] not in ['T','Z','X']
 assert not (Q/'L020_41/train_S.log').exists();prefix(g)
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid())))
 with (E/'watchdog.log').open('x') as f:w=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (E/'WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'WATCHDOG_READY.json').exists()
 old=same(g['old_controller']);assert old['state'] in ['S','R','D'];os.kill(old['pid'],signal.SIGSTOP)
 for _ in range(100):
  if same(g['old_controller'])['state']=='T':break
  time.sleep(.02)
 assert same(g['old_controller'])['state']=='T';assert same(g['child'])['state']!='Z' and same(g['training'])['state'] not in ['T','Z','X'];prefix(g)
 assert not (PREV/'L005_43_resume.exit').exists() and not (Q/'L005_41/evaluate.log').exists()
 put(E/'LIVE_COMMITTED.json',dict(time=time.time(),training_signals=0,existing_launcher=g['child'],existing_gpu=g['training'],old_coordinator=g['old_controller'],user_change='Start L02041 alongside L00543; priority L005 evaluation with serialized environments'))
 adopted=g['child'];launch('L020_41','train');put(E/'LIVE_LAUNCH.json',dict(time=time.time(),new_launcher=live['L020_41'][0].pid,unchanged_launcher=adopted['pid'],unchanged_training_pid=g['training']['pid'],training_signals=0))
 root_lock=None;failure=None
 while adopted:
  ch=same(adopted)
  if ch['state']=='Z':
   raw=ch['exit_code'];rc=raw>>8 if raw&127==0 else -(raw&127)
   if rc==0:complete('L005_43');prefix(g)
   else:failure=('L005_43',rc)
   put(E/'ADOPTED_L005_43_EXIT.json',dict(time=time.time(),kernel_exit=ch,returncode=rc,training_signals=0))
   old=same(g['old_controller']);assert old['state']=='T';os.kill(old['pid'],signal.SIGKILL)
   root_lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a')
   for i in range(100):
    try:fcntl.flock(root_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     if i==99:raise
     time.sleep(.1)
   with (PREV/'L005_43_resume.exit').open('x') as f:f.write(str(rc))
   put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),root_lock=True,source='existing seed43 recovery launcher natural kernel exit',returncode=rc,training_signals=0));adopted=None
  for case in list(live):
   rc=finish(case)
   if rc is not None:
    if rc==0:complete(case)
    else:failure=(case,rc)
  state('cross_lambda_training')
  if adopted:time.sleep(2)
 assert failure is None and root_lock is not None,failure
 # L005 evaluation is launched immediately at its natural completion, while the
 # already-started L02041 may continue training; environments share one lock.
 for seed in [41,42,43]:
  case='L005_'+str(seed);launch(case,'evaluate');assert finish(case,True)==0,(case,'evaluate')
  if 'L020_41' in live:
   rc=finish('L020_41')
   if rc is not None:assert rc==0;complete('L020_41')
 assert all(len(js(Q/('L005_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L005_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU',cross_lambda_training_authorized=True))
 train_pool(['L020_42','L020_43'])
 for seed in [41,42,43]:
  case='L020_'+str(seed);launch(case,'evaluate');assert finish(case,True)==0,(case,'evaluate')
 assert all(len(js(Q/('L020_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L020_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 state('complete');(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),training_signals=0))
  for case in list(live):
   try:finish(case,True)
   except BaseException:pass
  (E/'controller.exit').write_text('1');raise
