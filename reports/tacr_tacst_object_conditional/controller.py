"""Finite deferred OBJECT R / R+ST queues; exact same-seed41 AUC gates."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
from independent_audit import audit_run
Q=Path(__file__).resolve().parent
OLD=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def frozen():
 c=js(Q/'ORCHESTRATION_CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items()),'Frozen new contract changed'
def capacity(pending,old_pending=0):
 per=int(1.6*1024**3);need={'/':3*0,'/data':old_pending*per}
 for e in pending:need[e['volume']]+=e['storage_bound_bytes']
 out={}
 for m in ('/','/data'):
  v=os.statvfs(m);free=(v.f_bfree if m=='/data' else v.f_bavail)*v.f_frsize;reserve=(2 if m=='/data' else 1)*1024**3
  assert free>=need[m]+reserve,('RESOURCE_CAPACITY',m,free,need[m],reserve)
  out[m]=dict(free=free,pending_bound=need[m],reserve=reserve)
 return out
def wait_prior():
 assert sha(OLD/'ORCHESTRATION_CONTRACT.json')==js(Q/'PRIOR_QUEUE.json')['contract_sha256']
 while not (OLD/'controller.exit').exists():
  write(Q/'RUN_STATUS.json',dict(status='WAITING_PRIOR_QUEUE_NATURAL_EXIT',prior_queue=str(OLD),prior=js(OLD/'RUN_STATUS.json'),unix=time.time(),no_GPU_admission_yet=True));time.sleep(30)
 assert (OLD/'controller.exit').read_text().strip()=='0' and not (OLD/'FAILURE.json').exists(),'Prior queue engineering failure preserved; no blind takeover'
 c=js(OLD/'ORCHESTRATION_CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
 done=js(OLD/'RUN_STATUS.json')['completed'];proofs={}
 gate=js(OLD/'OBJECT_GATE.json');expected=['OBJECT_42','OBJECT_43']+(['GOAL_41','GOAL_42','GOAL_43'] if gate['passed'] else [])
 assert done==expected
 for key in done:
  r=js(OLD/(key+'.receipt.json'));e=next(e for e in js(OLD/'QUEUE.json') if e['key']==key);O=Path(e['output'])
  assert (OLD/(key+'.exit')).read_text().strip()=='0' and (O/'supervisor.exit').read_text().strip()=='0'
  assert js(O/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS' and r['acceptance_sha256']==sha(O/'PROCESS_EXIT_ACCEPTANCE.json')
  assert r['independent_audit_sha256']==sha(OLD/'independent'/(key+'.json')) and js(OLD/'independent'/(key+'.json'))['status']=='INDEPENDENT_FULL_MATRIX_LEDGER_PASS'
  proofs[key]=sha(OLD/(key+'.receipt.json'))
 write(Q/'PRIOR_PREFIX_ACCEPTANCE.json',dict(status='SERVER_PREFIX_ACCEPTED_LOCAL_AUDIT_STILL_REQUIRED',prior_contract_sha256=sha(OLD/'ORCHESTRATION_CONTRACT.json'),completed=done,receipts=proofs,old_gate_passed=gate['passed'],unix=time.time()))
def run(e,completed,pending):
 frozen();O=Path(e['output']);W=Q/e['key'];assert not O.exists() and not (W/'supervisor.pid.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU must be idle'
 cap=capacity(pending);write(Q/'RUN_STATUS.json',dict(status='PARTIAL',current=e['key'],completed=completed,unix=time.time(),capacity=cap))
 argv=['/data/mlr_ifa/env/bin/python','-u',str(Q/'seed_supervisor.py'),str(W)]
 with (Q/(e['key']+'.log')).open('x') as f:
  p=subprocess.Popen(argv,cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);write(Q/(e['key']+'.pid.json'),dict(pid=p.pid,argv=argv,unix=time.time()));rc=p.wait()
 (Q/(e['key']+'.exit')).write_text(str(rc)+'\n');assert rc==0,(e['key'],rc)
 result=audit_run(O,e['parent'],e['seed']);write(Q/'independent'/(e['key']+'.json'),result)
 write(Q/(e['key']+'.receipt.json'),dict(status='SERVER_INDEPENDENT_ACCEPTED_PENDING_LOCAL_AUDIT',key=e['key'],output=str(O),acceptance_sha256=sha(O/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(O/'CONTRACT.json'),independent_audit_sha256=sha(Q/'independent'/(e['key']+'.json')),orchestration_contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),unix=time.time()))
 completed.append(e['key']);frozen()
def main():
 assert not (Q/'controller.pid.json').exists();write(Q/'controller.pid.json',dict(pid=os.getpid(),unix=time.time()));rc=1;completed=[];locks=[]
 try:
  f=(Q/'queue.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f);frozen();wait_prior()
  for path in [OLD/'queue.lock',Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/queue.lock')]:
   f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  entries=js(Q/'QUEUE.json');pending=list(entries)
  for variant in ('TACR','TACRST'):
   group=[e for e in entries if e['variant']==variant];first=group[0];run(first,completed,pending);pending.remove(first)
   a=js(Q/'independent'/(first['key']+'.json'));actual=F(a['metrics']['AUC']['fraction']);baseline=F(a['baseline']['AUC']['fraction']);assert baseline==F(701,960)
   gate=dict(status='SERVER_EXACT_SAME_SEED41_GATE_PASS' if actual>baseline else 'SERVER_EXACT_SAME_SEED41_GATE_NOT_PASSED',variant=variant,seed=41,criterion='AUC_seed41_strictly_greater_than_RA_seed41',actual=str(actual),baseline=str(baseline),passed=actual>baseline,proof_sha256=sha(Q/'independent'/(first['key']+'.json')),unix=time.time(),local_acceptance_pending=True)
   write(Q/(variant+'_GATE.json'),gate)
   if gate['passed']:
    for e in group[1:]:run(e,completed,pending);pending.remove(e)
   else:
    for e in group[1:]:pending.remove(e)
    write(Q/(variant+'_EXPANSION_NOT_STARTED.json'),dict(reason='SEED41_AUC_NOT_STRICTLY_ABOVE_SAME_SEED_RA41',gate_sha256=sha(Q/(variant+'_GATE.json')),skipped=[e['key'] for e in group[1:]],unix=time.time()))
  rc=0;write(Q/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDIT',completed=completed,unix=time.time()))
 except Exception:
  write(Q/'FAILURE.json',dict(status='BLOCKED_ENGINEERING_EVIDENCE_PRESERVED_NO_RETRY',traceback=traceback.format_exc(),completed=completed,unix=time.time()));raise
 finally:(Q/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
