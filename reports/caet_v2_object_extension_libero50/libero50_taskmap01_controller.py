"""Evidence-preserving task-name resolution recovery; original queue remains stopped."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;Q=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
 lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'controller.pid.json').exists(),'Explicit new version required after failure'
 write(D/'controller.pid.json',dict(pid=os.getpid(),unix=time.time()))
 done=['CAETv2_OBJECT_41','CAETv2_OBJECT_43'];rc=1
 try:
  c=json.loads((D/'CONTRACT.json').read_text())
  def verify():
   for p,h in c['files'].items():assert sha(p)==h,p
   assert (Q/'controller.exit').read_text().strip()=='1'
   assert (Q/'capacity_recovery_v1/watcher.exit').read_text().strip()=='1'
  verify()
  for seed in (41,43):
   receipt=json.loads((Q/('CAETv2_OBJECT_%d.receipt.json'%seed)).read_text());o=Path(receipt['output'])
   assert receipt['acceptance_sha256']==sha(o/'PROCESS_EXIT_ACCEPTANCE.json')
   assert receipt['contract_sha256']==sha(o/'CONTRACT.json')
  assert json.loads((D/'MAPPING_AUDIT.json').read_text())['status']=='PASS_50_NATIVE_IDENTITIES'
  entries=json.loads((D/'QUEUE.json').read_text());assert [e['seed'] for e in entries]==[41,42,43]
  for e in entries:
   verify()
   assert not Path(e['output']).exists()
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
   st=os.statvfs(e['storage_mount']);normal=st.f_bavail*st.f_frsize;all_free=st.f_bfree*st.f_frsize
   if e['storage_mount']=='/data':assert os.geteuid()==0 and all_free>=4500000000+2*1024**3,('RESOURCE_CAPACITY',all_free)
   else:assert normal>=4500000000,('RESOURCE_CAPACITY',normal)
   if e['seed']==41:
    st=os.statvfs('/data');assert st.f_bfree*st.f_frsize>=2*1024**3+2000000000,'Shared pretraining capacity'
   write(D/'RUN_STATUS.json',dict(status='PARTIAL_TASKMAP01',current=e['key'],completed=done,unix=time.time(),normal_free=normal,root_free=all_free))
   with (D/(e['key']+'.log')).open('x') as f:
    p=subprocess.Popen(e['argv'],cwd=D,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    write(D/(e['key']+'.pid.json'),dict(pid=p.pid,argv=e['argv'],unix=time.time()));code=p.wait()
   (D/(e['key']+'.exit')).write_text(str(code)+'\n');assert code==0,(e['key'],code)
   out=Path(e['output']);a=json.loads((out/'PROCESS_EXIT_ACCEPTANCE.json').read_text())
   assert a['status']=='PASS' and (out/'supervisor.exit').read_text().strip()=='0' and (out/'pipeline.exit').read_text().strip()=='0'
   write(D/(e['key']+'.receipt.json'),dict(status='SERVER_ACCEPTED_AWAITING_LOCAL_AUDIT',key=e['key'],seed=e['seed'],method='RA_IFA',output=str(out),acceptance_sha256=sha(out/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(out/'CONTRACT.json'),unix=time.time()))
   done.append(e['key'])
  rc=0;write(D/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDITS',completed=done,unix=time.time()))
 except Exception:
  write(D/'FAILURE.json',dict(status='BLOCKED_ENGINEERING_NO_BLIND_RETRY',traceback=traceback.format_exc(),completed=done,unix=time.time()));raise
 finally:(D/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
