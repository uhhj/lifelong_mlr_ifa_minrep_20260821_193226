"""Install only this reviewed CAET-v2 source; finite authorized gate2→gate3."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
W='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021'
SCRIPT=r'''
import os,json,hashlib,shutil,subprocess,time
from pathlib import Path
W=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021')
Q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
assert (Q/'COVR_CANCELLED_BY_USER.json').is_file()
assert not (W/'source').exists() and not (W/'supervisor.pid.json').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,('GPU already occupied',gpu)
assert shutil.disk_usage('/data').free>2*1024**3
bundle=json.loads((W/'DEPLOY_SOURCE_BUNDLE.json').read_text())
source=Q/'workers/CAET_OBJECT_42/source'
shutil.copytree(source,W/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name,item in bundle.items():
 p=(W/'source'/name).resolve();assert (W/'source').resolve() in p.parents and p.suffix=='.py'
 raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(raw)
receipt=dict(installed=time.time(),source_files=len(bundle),gpu_processes_before=gpu,data_free=shutil.disk_usage('/data').free,system_free=shutil.disk_usage('/').free)
(W/'INSTALL_RECEIPT.json').write_text(json.dumps(receipt)+'\n')
with (W/'supervisor.log').open('x') as log:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(W/'supervisor.py')],cwd=W,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt['supervisor_pid']=p.pid
(W/'DISPATCH_RECEIPT.json').write_text(json.dumps(receipt)+'\n')
print(json.dumps(receipt))
'''

def main():
 gate=R/'gate1_20261021_200229';assert json.loads((gate/'LOCAL_AUDIT.json').read_text())['status']=='GATE1_PASS'
 assert json.loads((gate/'EVIDENCE.json').read_text())['protocol_sha256']==hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest()
 bundle={p.relative_to(R/'additions').as_posix():dict(text=p.read_text(encoding='utf8'),sha256=hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest()) for p in (R/'additions').rglob('*.py')}
 stamp=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));stamp.mkdir()
 (stamp/'SOURCE_MANIFEST.json').write_text(json.dumps({n:dict(sha256=v['sha256'],bytes=len(v['text'].encode())) for n,v in bundle.items()},indent=2)+'\n',encoding='utf8')
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  s=c.open_sftp();s.mkdir(W)
  uploads={'DEPLOY_SOURCE_BUNDLE.json':json.dumps(bundle).encode(),'PROTOCOL.md':(R/'PROTOCOL.md').read_bytes(),'GATE1_EVIDENCE.json':(gate/'EVIDENCE.json').read_bytes(),'GATE1_LOCAL_AUDIT.json':(gate/'LOCAL_AUDIT.json').read_bytes(),'IMPLEMENTATION_NOTES.md':(R/'IMPLEMENTATION_NOTES.md').read_bytes(),'supervisor.py':(R/'supervisor.py').read_bytes()}
  for name,raw in uploads.items():
   with s.open(W+'/'+name,'wb') as f:f.write(raw)
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (stamp/'DEPLOY_ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stdout=raw.decode(),stderr=err.decode(),uploads={n:hashlib.sha256(b).hexdigest() for n,b in uploads.items()}),indent=2)+'\n',encoding='utf8')
  assert rc==0,err.decode();print(raw.decode());print(stamp)
 finally:pw=None;c.close()

if __name__=='__main__':main()
