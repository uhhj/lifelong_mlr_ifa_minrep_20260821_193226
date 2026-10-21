"""Versioned capacity-only handoff. Waits for old controller EXIT, never kills it.
Uses existing root-writable DATA free blocks, preserving 2 GiB emergency space.
No filesystem tuning, source changes, deletion, scientific retries or GPU prechecks.
"""
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
Q=Path(__file__).resolve().parent
D=Q/'capacity_recovery_v1'
RESERVE=2*1024**3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def allowed(entry,capacity,output_exists,old_exit,failure):
 return (old_exit==1 and 'RESOURCE_CAPACITY' in failure
  and entry['key'] in ('RA_IFA_LIBERO50_42','RA_IFA_LIBERO50_43')
  and entry['storage_mount']=='/data' and not output_exists and capacity['euid']==0
  and capacity['free_all_blocks']>=entry['minimum_free_bytes']+RESERVE)
def capacity():
 st=os.statvfs('/data')
 return dict(euid=os.geteuid(),free_all_blocks=st.f_bfree*st.f_frsize,
  available_unprivileged=st.f_bavail*st.f_frsize,emergency_reserve_bytes=RESERVE)
def verify_contract():
 sup=json.loads((D/'CONTRACT.json').read_text())
 assert sha(Q/'ORCHESTRATION_CONTRACT.json')==sup['original_orchestration_sha256']
 assert sha(__file__)==sup['recovery_script_sha256']
 for p,h in json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())['files'].items():assert sha(p)==h,p
 return sup
def main():
 lock=(D/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'watcher.pid.json').exists()
 verify_contract();write(D/'watcher.pid.json',dict(pid=os.getpid(),unix=time.time()))
 write(D/'RUN_STATUS.json',dict(status='ARMED_WAITING_ORIGINAL_EXIT',unix=time.time(),does_not_modify_running_jobs=True))
 rc=1
 try:
  while not (Q/'controller.exit').exists():time.sleep(30)
  old_exit=int((Q/'controller.exit').read_text())
  if old_exit==0:
   write(D/'RUN_STATUS.json',dict(status='NO_RECOVERY_NEEDED_ORIGINAL_COMPLETE',unix=time.time()));rc=0;return
  failure=json.loads((Q/'FAILURE.json').read_text())
  assert old_exit==1 and 'RESOURCE_CAPACITY' in failure['traceback'],'Non-capacity failure requires its own evidence-preserving recovery'
  # OS lock is released only when the original controller has exited.
  qlock=(Q/'queue.lock').open('a');fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify_contract();entries=json.loads((Q/'QUEUE.json').read_text());completed=failure['completed']
  assert completed==[x['key'] for x in entries[:len(completed)]]
  for e in entries[:len(completed)]:
   r=json.loads((Q/(e['key']+'.receipt.json')).read_text());out=Path(e['output'])
   assert r['key']==e['key'] and r['acceptance_sha256']==sha(out/'PROCESS_EXIT_ACCEPTANCE.json')
   assert (Q/(e['key']+'.exit')).read_text().strip()=='0'
  write(D/'HANDOFF.json',dict(unix=time.time(),old_exit=old_exit,old_failure_sha256=sha(Q/'FAILURE.json'),completed=completed,scope='capacity accounting only; no stopped worker is resumed',filesystem_settings_changed=False,files_deleted=False))
  for e in entries[len(completed):]:
   verify_contract();cap=capacity()
   assert allowed(e,cap,Path(e['output']).exists(),old_exit,failure['traceback']),('INSUFFICIENT_SAFE_CAPACITY_OR_UNSUPPORTED_ENTRY',e['key'],cap)
   assert not (Q/(e['key']+'.exit')).exists(),'Original entry already attempted'
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
   write(D/'RUN_STATUS.json',dict(status='PARTIAL_CAPACITY_RECOVERY',current=e['key'],completed=completed,capacity=cap,unix=time.time()))
   with (D/(e['key']+'.log')).open('x') as log:
    p=subprocess.Popen(e['argv'],cwd=Q,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    write(D/(e['key']+'.pid.json'),dict(pid=p.pid,argv=e['argv'],unix=time.time()));code=p.wait()
   (D/(e['key']+'.exit')).write_text(str(code)+'\n');assert code==0,(e['key'],code)
   out=Path(e['output']);accepted=json.loads((out/'PROCESS_EXIT_ACCEPTANCE.json').read_text())
   assert accepted['status']=='PASS' and (out/'supervisor.exit').read_text().strip()=='0' and (out/'pipeline.exit').read_text().strip()=='0'
   write(D/(e['key']+'.receipt.json'),dict(status='SERVER_ACCEPTED_AWAITING_LOCAL_AUDIT',key=e['key'],seed=e['seed'],method=e['method'],output=str(out),acceptance_sha256=sha(out/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(out/'CONTRACT.json'),unix=time.time()))
   completed.append(e['key'])
  write(D/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDITS',completed=completed,unix=time.time()));rc=0
 except Exception:
  write(D/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),status='BLOCKED_ENGINEERING_NO_BLIND_RETRY'));raise
 finally:(D/'watcher.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
