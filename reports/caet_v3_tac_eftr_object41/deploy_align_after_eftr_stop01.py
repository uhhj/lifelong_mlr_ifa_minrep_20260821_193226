"""Deploy an external one-entry continuation; keep original contracts untouched."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import sys,os,json,time,hashlib,subprocess,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01');D=Q/'align_after_eftr_stop01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.load(sys.stdin);assert not D.exists()
assert (Q/'controller.exit').read_text().strip()=='1'
t=Q/'eftr_user_stop01/TERMINATION_ACCEPTANCE.json';stop=json.loads(t.read_text());assert stop['status']=='EARLY_TERMINATED_BY_USER' and not stop['processes_after']
original=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text());assert all(sha(p)==h for p,h in original['files'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd:assert str(Q) not in cmd and str(Q) not in cwd,('residual',p.name)
 except OSError:pass
entry=next(e for e in json.loads((Q/'QUEUE.json').read_text()) if e['key']=='ALIGN_OBJECT_41')
assert not Path(entry['output']).exists() and not (Path(entry['argv'][-1])/'supervisor.pid.json').exists()
prefix=Q/'TAC_OBJECT_41.receipt.json';assert json.loads(prefix.read_text())['status']=='SERVER_ACCEPTED_AWAITING_LOCAL_AUDIT'
lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
compile(d['source'],'controller.py','exec');D.mkdir();(D/'controller.py').write_text(d['source']);(D/'ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n')
files={str(p):sha(p) for p in [D/'controller.py',D/'ENTRY.json',t,prefix,Q/'ORCHESTRATION_CONTRACT.json']}
(D/'CONTRACT.json').write_text(json.dumps(dict(status='FROZEN',files=files,user_scope='Stop EFTR with preserved evidence, continue original ALIGN only',original_science_unchanged=True,unix=time.time()),indent=2)+'\n')
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (D/'controller.log').open('x') as log:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(D/'controller.py')],cwd=D,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='DISPATCHED_AWAITING_ALIGN_OWN_GPU_ADMISSION',pid=p.pid,directory=str(D),entry=entry,contract_sha256=sha(D/'CONTRACT.json'),original_frozen_hash_count=len(original['files']),original_frozen_hashes='PASS',termination_sha256=sha(t),unix=time.time())
(D/'DISPATCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
def main():
 out=R/('align_after_stop01_deploy_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 source=(R/'align_after_eftr_stop01_controller.py').read_text(encoding='utf8');compile(source,'controller.py','exec')
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);i.write(json.dumps(dict(source=source)));i.flush();i.channel.shutdown_write()
  raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (out/'DISPATCH_RECEIPT.json').write_bytes(raw);print(out);print(raw.decode())
 finally:pw=None;c.close()
if __name__=='__main__':main()
