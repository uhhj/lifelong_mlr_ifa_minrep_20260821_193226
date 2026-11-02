"""Local-only construction of live adoption; never starts a scientific process."""
import ast,json
from pathlib import Path
E=Path(__file__).resolve().parent;P=E.parent/'engineering_parallel_3090_20261102_v1'
s=(P/'controller.py').read_text(encoding='utf8').split('def main():')[0]
s=s.replace('live={}','live={};adopted=None')
s=s.replace("E.name+'/CONTRACT.json'", "'engineering_parallel_3090_20261102_v1/CONTRACT.json',E.name+'/CONTRACT.json'")
s=s.replace(" put(E/'STATUS.json',x);put(Q/'STATUS.json',x)"," if adopted:x['active']['L005_42']=dict(pid=adopted['pid'],phase='train',adopted_existing_process=True)\n assert len(x['active'])<=2\n put(E/'STATUS.json',x);put(Q/'STATUS.json',x)")
s+='''def same(identity):
 actual=proc(identity['pid']);assert actual['start']==identity['start'];return actual
def complete(case):
 r=Q/case;t=js(r/'TRAIN_S.json');assert t['epochs']==400 and t['steps']==105500 and t['physical_optimizer_calls']==195300 and t['training_collection_episodes']==40
 assert len(js(r/'EPOCHS_S.json'))==400
 for st in range(6,10):assert (r/('collect_S_%d.exit'%st)).read_text().strip()=='0' and js(r/('MEMORY_S_%d.json'%st))['training_attempts']==10
def main():
 global adopted
 own=(E/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (E/'STARTED.json').exists();verify();g=js(E/'GUARD.json')
 assert js(Q/'engineering_parallel_3090_20261102_v1/PARALLEL_ADMISSION.json')['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
 assert js(E/'STARTUP_LOCK_VALIDATION.json')['status']=='PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK'
 assert same(g['controllers']['root'])['state']=='T'
 assert same(g['controllers']['parallel'])['state'] in ['S','R','D'] and same(g['controllers']['watchdog'])['state'] in ['S','R','D']
 assert same(g['child'])['state']!='Z' and same(g['training'])['state'] not in ['T','Z','X']
 assert not (Q/'L005_43/train_S.log').exists();prefix_guard(g)
 put(E/'STARTED.json',dict(time=time.time(),pid=os.getpid(),identity=proc(os.getpid())))
 with (E/'watchdog.log').open('x') as f:w=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (E/'WATCHDOG_READY.json').exists():break
  time.sleep(.1)
 assert (E/'WATCHDOG_READY.json').exists()
 for key in ['watchdog','parallel']:
  old=same(g['controllers'][key]);assert old['state'] in ['S','R','D'];os.kill(old['pid'],signal.SIGSTOP)
 for _ in range(50):
  if all(same(g['controllers'][key])['state']=='T' for key in ['root','parallel','watchdog']):break
  time.sleep(.02)
 for key in ['root','parallel','watchdog']:assert same(g['controllers'][key])['state']=='T'
 current=js(Q/'L005_42/TRAIN_STATUS.json');assert current['phase']=='train' and current['pid']==g['training']['pid'] and current['epoch_complete']<96
 assert same(g['training'])['state'] not in ['T','Z','X'];prefix_guard(g)
 (E/'HOOK_ENABLED.json').write_text('{}')
 put(E/'LIVE_COMMITTED.json',dict(time=time.time(),training_signals=0,root_lock_owner=g['controllers']['root']['pid'],root_lock_state='held by frozen original coordinator until adopted launcher exits',future_owner=os.getpid(),current_launcher=g['child'],startup_lock_enabled=True))
 adopted=g['child'];launch('L005_43','train');put(E/'LIVE_LAUNCH.json',dict(time=time.time(),seed43_launcher=live['L005_43'][0].pid,seed42_launcher=adopted['pid'],seed42_gpu=g['training']['pid'],training_signals=0,max_training_concurrency=2))
 failure=None;root_lock=None
 while adopted is not None or live:
  if adopted is not None:
   ch=same(adopted)
   if ch['state']=='Z':
    raw=ch['exit_code'];rc=raw>>8 if raw&127==0 else -(raw&127)
    if rc==0:complete('L005_42');prefix_guard(g)
    else:failure=('L005_42',rc)
    put(E/'ADOPTED_L005_42_EXIT.json',dict(time=time.time(),kernel_exit=ch,returncode=rc,training_signals=0))
    # Only already-frozen coordinators are ended, after seed42 exits naturally.
    for key in ['watchdog','parallel','root']:
     old=same(g['controllers'][key]);assert old['state']=='T';os.kill(old['pid'],signal.SIGKILL)
    root_lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a')
    for attempt in range(100):
     try:fcntl.flock(root_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:
      if attempt==99:raise
      time.sleep(.1)
    with (Q/'L005_42/train_S.exit').open('x') as f:f.write(str(rc))
    put(E/'HANDOFF_COMPLETE.json',dict(time=time.time(),root_lock=True,source='existing seed42 launcher natural exit',returncode=rc,training_signals=0));adopted=None
  for case in list(live):
   rc=finish(case)
   if rc is not None:
    if rc==0:complete(case)
    else:failure=(case,rc)
  state('parallel_L005_live_adoption')
  if adopted is not None or live:time.sleep(2)
 assert failure is None and root_lock is not None,failure
 for seed in [41,42,43]:one('L005_'+str(seed),'evaluate')
 assert all(len(js(Q/('L005_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L005_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 train_pool(['L020_41','L020_42','L020_43'])
 for seed in [41,42,43]:one('L020_'+str(seed),'evaluate')
 assert all(len(js(Q/('L020_'+str(seed))/'OUTCOMES.json'))==200 for seed in [41,42,43]);put(Q/'L020_EVALUATION_COMPLETE.json',dict(time=time.time(),episodes=600,status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU'))
 state('complete');(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),training_signals=0,already_started_peers='left running; drain owned child before coordinator exit'))
  for case in list(live):
   try:finish(case,True)
   except BaseException:pass
  (E/'controller.exit').write_text('1');raise
'''
ast.parse(s);(E/'controller.py').write_text(s,encoding='utf8')
watch='''import json,os,time,signal
from pathlib import Path
E=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text())
def proc(i):
 p=Path('/proc')/str(i['pid'])
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split();return dict(state=s[0],start=s[19])
g=js(E/'GUARD.json');owner=js(E/'STARTED.json')['identity'];(E/'WATCHDOG_READY.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
while True:
 if (E/'HANDOFF_COMPLETE.json').exists():break
 p=proc(owner)
 if p is None or p['start']!=owner['start'] or p['state']=='Z':
  if (E/'LIVE_COMMITTED.json').exists():
   (E/'OWNER_LOST_AFTER_LIVE_COMMIT.json').write_text(json.dumps(dict(time=time.time(),action='Keep old coordinators frozen; never duplicate seed43; no training signals',training_signals=0)))
  else:
   flag=E/'HOOK_ENABLED.json'
   if flag.exists():assert flag.read_text()=='{}';flag.unlink()
   for key in ['parallel','watchdog']:
    old=g['controllers'][key];p=proc(old)
    if p and p['start']==old['start'] and p['state']=='T':os.kill(old['pid'],signal.SIGCONT)
   (E/'WATCHDOG_RELEASE.json').write_text(json.dumps(dict(time=time.time(),action='Restore prior waiting coordinator/watchdog only; disable new startup lock',training_signals=0)))
  break
 time.sleep(2)
'''
ast.parse(watch);(E/'watchdog.py').write_text(watch,encoding='utf8')
wrapper='''import sys,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;case=sys.argv[1]
assert case in ['L005_43','L020_41','L020_42','L020_43']
c=json.loads((E/'CONTRACT.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
assert (E/'LIVE_COMMITTED.json').exists() and (E/'HOOK_ENABLED.json').exists()
P=Q/'engineering_checks_3090_20261102_v1/train_wrapper.py'
exec(compile(P.read_text(),str(P),'exec'),dict(__file__=str(P),__name__='__main__'))
'''
ast.parse(wrapper);(E/'train_wrapper.py').write_text(wrapper,encoding='utf8')
print(json.dumps(dict(status='LOCAL_ONLY_NOT_DEPLOYED',current_training_signals=0,immediate_start='L005_43',adopt_existing='L005_42',root_lock_handoff='only after existing launcher natural exit',collection_startup_hook='exact-path separate interpreter before GPU/env',scientific_changes=False)))
