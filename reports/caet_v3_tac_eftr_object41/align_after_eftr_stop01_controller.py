"""Single frozen ALIGN entry after explicit user termination of EFTR."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent
Q=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
 locks=[]
 for q in (Q,Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022'),Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_AUX8_OBJECT41_20261022')):
  f=(q/'queue.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not (D/'controller.pid.json').exists(),'No implicit restart'
 write(D/'controller.pid.json',dict(pid=os.getpid(),unix=time.time()))
 rc=1
 try:
  c=json.loads((D/'CONTRACT.json').read_text())
  assert all(sha(p)==h for p,h in c['files'].items())
  old=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text());assert all(sha(p)==h for p,h in old['files'].items())
  termination=json.loads((Q/'eftr_user_stop01/TERMINATION_ACCEPTANCE.json').read_text())
  assert termination['status']=='EARLY_TERMINATED_BY_USER' and not termination['processes_after']
  assert (Q/'controller.exit').read_text().strip()=='1'
  entry=json.loads((D/'ENTRY.json').read_text());assert entry['key']=='ALIGN_OBJECT_41'
  assert entry==next(x for x in json.loads((Q/'QUEUE.json').read_text()) if x['key']=='ALIGN_OBJECT_41')
  out=Path(entry['output']);worker=Path(entry['argv'][-1])
  assert not out.exists() and not (worker/'supervisor.pid.json').exists()
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  v=os.statvfs(entry['storage_mount']);free=v.f_bfree*v.f_frsize-2*1024**3
  assert free>=entry['minimum_free_bytes'],('RESOURCE_CAPACITY',free)
  write(D/'RUN_STATUS.json',dict(status='PARTIAL',current=entry['key'],completed_prefix=['TAC_OBJECT_41'],terminated=['EFTR_OBJECT_41'],unix=time.time(),disk_free_after_reserve=free))
  with (D/'ALIGN_OBJECT_41.log').open('x') as log:
   p=subprocess.Popen(entry['argv'],cwd=Q,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   write(D/'ALIGN_OBJECT_41.pid.json',dict(pid=p.pid,argv=entry['argv'],unix=time.time()))
   code=p.wait()
  (D/'ALIGN_OBJECT_41.exit').write_text(str(code)+'\n');assert code==0,code
  accept=json.loads((out/'PROCESS_EXIT_ACCEPTANCE.json').read_text());assert accept['status']=='PASS'
  assert (out/'supervisor.exit').read_text().strip()=='0' and (out/'pipeline.exit').read_text().strip()=='0'
  write(D/'ALIGN_OBJECT_41.receipt.json',dict(status='SERVER_ACCEPTED_AWAITING_LOCAL_AUDIT',key=entry['key'],seed=entry['seed'],method=entry['method'],output=str(out),acceptance_sha256=sha(out/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(out/'CONTRACT.json'),continuation_contract_sha256=sha(D/'CONTRACT.json'),unix=time.time()))
  rc=0;write(D/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDIT',completed=['ALIGN_OBJECT_41'],terminated=['EFTR_OBJECT_41'],unix=time.time()))
 except Exception:
  write(D/'FAILURE.json',dict(status='BLOCKED_REQUIRES_EVIDENCE_PRESERVING_RECOVERY',traceback=traceback.format_exc(),unix=time.time()));raise
 finally:(D/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
