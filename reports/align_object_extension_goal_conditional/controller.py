"""Finite OBJECT42/43 then exact mean-AUC gate then GOAL41/42/43."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
from independent_audit import audit_run
Q=Path(__file__).resolve().parent
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen():
 c=js(Q/'ORCHESTRATION_CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
def run(e,completed):
 frozen();O=Path(e['output']);W=Q/e['key'];assert not O.exists() and not (W/'supervisor.pid.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 v=os.statvfs('/data');free=v.f_bfree*v.f_frsize;assert free>=e['storage_bound_bytes']+2*1024**3,('RESOURCE_CAPACITY',free,e['storage_bound_bytes'])
 write(Q/'RUN_STATUS.json',dict(status='PARTIAL',current=e['key'],completed=completed,unix=time.time(),disk_root_free=free))
 argv=['/data/mlr_ifa/env/bin/python','-u',str(Q/'seed_supervisor.py'),str(W)]
 with (Q/(e['key']+'.log')).open('x') as f:
  p=subprocess.Popen(argv,cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);write(Q/(e['key']+'.pid.json'),dict(pid=p.pid,argv=argv,unix=time.time()));rc=p.wait()
 (Q/(e['key']+'.exit')).write_text(str(rc)+'\n');assert rc==0,(e['key'],rc)
 result=audit_run(O,e['parent'],e['seed'],e['suite']);write(Q/'independent'/ (e['key']+'.json'),result)
 write(Q/(e['key']+'.receipt.json'),dict(status='SERVER_INDEPENDENT_ACCEPTED_PENDING_LOCAL_AUDIT',key=e['key'],output=str(O),acceptance_sha256=sha(O/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(O/'CONTRACT.json'),independent_audit_sha256=sha(Q/'independent'/(e['key']+'.json')),orchestration_contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),unix=time.time()))
 completed.append(e['key']);frozen()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def aggregate(suite,entries):
 values={};base={};proofs={}
 for e in entries:
  key=e['key'];a=js(Q/'independent'/(key+'.json'));assert a['status']=='INDEPENDENT_FULL_MATRIX_LEDGER_PASS'
  values[e['seed']]=F(a['metrics']['AUC']['fraction']);base[e['seed']]=F(a['baseline']['AUC']['fraction']);proofs[key]=sha(Q/'independent'/(key+'.json'))
 mean=sum(values.values())/3;bm=sum(base.values())/3
 return dict(status='SERVER_EXACT_GATE_PASS' if mean>bm else 'SERVER_EXACT_GATE_NOT_PASSED',suite=suite,seeds=[41,42,43],criterion='equal_seed_mean_AUC_strictly_greater_than_same_seed_RA',auc={str(k):str(v) for k,v in values.items()},baseline_auc={str(k):str(v) for k,v in base.items()},mean_auc=str(mean),mean_baseline_auc=str(bm),delta=str(mean-bm),passed=mean>bm,proofs=proofs,unix=time.time(),local_acceptance_pending=True)
def main():
 assert not (Q/'controller.pid.json').exists();write(Q/'controller.pid.json',dict(pid=os.getpid(),unix=time.time()));rc=1;completed=[];locks=[]
 try:
  for path in [Q/'queue.lock',Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/queue.lock')]:
   f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  frozen();entries=js(Q/'QUEUE.json');prefix=js(Q/'ACCEPTED_OBJECT41.json')
  a=audit_run(prefix['output'],prefix['parent'],41,'OBJECT');assert a['metrics']==prefix['local_final_audit']['metrics'];assert a['acceptance_sha256']==prefix['acceptance_sha256']
  write(Q/'independent/OBJECT_41.json',a)
  for e in entries[:2]:run(e,completed)
  gate=aggregate('OBJECT',[prefix]+entries[:2]);write(Q/'OBJECT_GATE.json',gate)
  if gate['passed']:
   for e in entries[2:]:run(e,completed)
   write(Q/'GOAL_AGGREGATE.json',aggregate('GOAL',entries[2:]))
  else:write(Q/'GOAL_NOT_STARTED.json',dict(reason='OBJECT_MEAN_AUC_NOT_STRICTLY_HIGHER',gate_sha256=sha(Q/'OBJECT_GATE.json'),unix=time.time()))
  rc=0;write(Q/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDIT',completed=completed,object_gate_passed=gate['passed'],unix=time.time()))
 except Exception:
  write(Q/'FAILURE.json',dict(status='BLOCKED_ENGINEERING_EVIDENCE_PRESERVED_NO_RETRY',traceback=traceback.format_exc(),completed=completed,unix=time.time()));raise
 finally:(Q/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
