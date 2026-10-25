"""Engineering02: inherit natural D exit, then execute O/S once."""
import os,subprocess,json,time,fcntl,traceback,signal,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
def write(n,x):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return dict(state=fields[0],starttime=fields[19],exit_code=int(fields[49]),stat_sha256=hashlib.sha256(raw.encode()).hexdigest())
def main():
 lock=open('/data/mlr_ifa/CCBP_V4_FULL_GOAL41_20261025_engineering02.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'STARTED.json').exists();write('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 tr=json.loads((R/'TRANSITION_INPUT.json').read_text());old=Path(tr['old_root']);cp=tr['old_controller'];worker=tr['old_torchrun']
 write('STATUS.json',dict(phase='await_existing_D',controller=os.getpid(),existing_D_pid=worker,time=time.time(),no_D_restart=True))
 while True:
  ctrl=proc(cp);assert ctrl['starttime']==tr['controller_starttime'] and ctrl['state'] in ['T','t'],'Old dispatch must remain paused'
  st=proc(worker);assert st['starttime']==tr['worker_starttime'],'PID reuse'
  if st['state']=='Z':break
  time.sleep(10)
 assert st['exit_code']==0,('Original D failed',st)
 hist=json.loads((old/'EPOCHS_D.json').read_text());summary=json.loads((old/'TRAIN_D.json').read_text());assert len(hist)==400 and hist[-1]['steps']==76100 and summary['epochs']==400 and summary['steps']==76100
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 inherited={}
 for p in old.glob('*'):
  if not p.is_file():continue
  if (p.suffix=='.json' and ('_D_' in p.name or p.stem.endswith('_D'))) or (p.suffix=='.exit' and p.name.startswith('collect_D_')):
   target=R/p.name
   if target.exists():assert sha(target)==sha(p)
   else:shutil.copyfile(p,target)
   inherited[p.name]=sha(p)
 for t in range(6,10):
  assert (old/('collect_D_%d.exit'%t)).read_text()=='0'
  for prefix in ['STAGE','MEMORY','COLLECTION_STATE_PARITY']:assert (R/('%s_D_%d.json'%(prefix,t))).exists()
 events=(old/'COLLECTION_EVENTS.jsonl').read_bytes();assert all(json.loads(s)['arm']=='D' for s in events.splitlines())
 (R/'COLLECTION_EVENTS.jsonl').write_bytes(events);(R/'INHERITED_D_COLLECTION_EVENTS.jsonl').write_bytes(events)
 write('INHERITED_D_EXIT_OBSERVATION.json',dict(status='PASS_OBSERVED_NATURAL_D_EXIT',worker=worker,wait_status=st,epochs=400,steps=76100,files=inherited,source_contract=sha(old/'CONTRACT.json'),completed_epochs_repeated=0,collection_attempts_repeated=0,time=time.time()))
 (R/'train_D.exit').write_text('0')
 assert proc(cp)['starttime']==tr['controller_starttime'];os.kill(cp,signal.SIGKILL)
 write('DISPATCHER_RETIRED.json',dict(old_controller=cp,reason='D completed naturally before versioned short-history repair handoff',time=time.time(),old_controller_not_claimed_exit0=True))
 jobs=[('train_'+a,[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(R/'train.py'),a]) for a in 'OS']+[('evaluate',[PY,'-B',str(R/'evaluate.py')])]
 for name,argv in jobs:
  with (R/(name+'.log')).open('x') as f:
   p=subprocess.Popen(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'));write('STATUS.json',dict(phase=name,pid=p.pid,time=time.time(),controller=os.getpid()));rc=p.wait()
  (R/(name+'.exit')).write_text(str(rc));assert rc==0,(name,rc)
 write('STATUS.json',dict(phase='complete',time=time.time(),controller=os.getpid()));(R/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));(R/'controller.exit').write_text('1');raise
