"""One finite explicit allowlist. No scientific retries or cancelled methods."""
import fcntl,json,os,subprocess,time,traceback,hashlib,shutil
from pathlib import Path
Q=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
 lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (Q/'controller.pid.json').exists(),'No implicit restart'
 write(Q/'controller.pid.json',dict(pid=os.getpid(),unix=time.time()))
 entries=json.loads((Q/'QUEUE.json').read_text());contract=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
 rc=1;done=[]
 try:
  assert [e['key'] for e in entries]==['CAETv2_OBJECT_41','CAETv2_OBJECT_43','RA_IFA_LIBERO50_41','RA_IFA_LIBERO50_42','RA_IFA_LIBERO50_43']
  for e in entries:
   for path,h in contract['files'].items():assert sha(path)==h,path
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
   free=shutil.disk_usage(e['storage_mount']).free
   assert free>=e['minimum_free_bytes'],('RESOURCE_CAPACITY',e['key'],free,e['minimum_free_bytes'])
   write(Q/'RUN_STATUS.json',dict(status='PARTIAL',current=e['key'],completed=done,unix=time.time(),disk_free=free))
   with (Q/(e['key']+'.log')).open('x') as f:
    p=subprocess.Popen(e['argv'],cwd=Q,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    write(Q/(e['key']+'.pid.json'),dict(pid=p.pid,unix=time.time(),argv=e['argv']))
    code=p.wait()
   (Q/(e['key']+'.exit')).write_text(str(code)+'\n');assert code==0,(e['key'],code)
   out=Path(e['output']);accept=json.loads((out/'PROCESS_EXIT_ACCEPTANCE.json').read_text())
   assert accept['status']=='PASS'
   assert (out/'supervisor.exit').read_text().strip()=='0' and (out/'pipeline.exit').read_text().strip()=='0'
   write(Q/(e['key']+'.receipt.json'),dict(status='SERVER_ACCEPTED_AWAITING_LOCAL_AUDIT',key=e['key'],seed=e['seed'],method=e['method'],output=str(out),acceptance_sha256=sha(out/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(out/'CONTRACT.json'),unix=time.time()))
   done.append(e['key'])
  rc=0;write(Q/'RUN_STATUS.json',dict(status='SERVER_FINISHED_AWAITING_LOCAL_AUDITS',completed=done,unix=time.time()))
 except Exception:
  write(Q/'FAILURE.json',dict(status='BLOCKED_REQUIRES_EVIDENCE_PRESERVING_RECOVERY',traceback=traceback.format_exc(),completed=done,unix=time.time()))
  raise
 finally:(Q/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
