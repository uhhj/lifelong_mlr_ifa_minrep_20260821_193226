import getpass,json,shlex,hashlib,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent;W='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021'
script=r'''
import subprocess,json,time
from pathlib import Path
W=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021')
assert not (W/'recovery_environment01.log').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with (W/'recovery_environment01.log').open('x') as log:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(W/'recovery_environment01.py')],cwd=W,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,time=time.time())))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 s=c.open_sftp();raw=(R/'recovery_environment01.py').read_bytes()
 try:s.stat(W+'/recovery_environment01.py');raise RuntimeError('Recovery already installed')
 except FileNotFoundError:pass
 with s.open(W+'/recovery_environment01.py','wb') as f:f.write(raw)
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(script),timeout=30);result=o.read();err=e.read();rc=o.channel.recv_exit_status()
 receipt=dict(exit=rc,stdout=result.decode(),stderr=err.decode(),script_sha256=hashlib.sha256(raw).hexdigest(),time=datetime.datetime.now().astimezone().isoformat())
 (R/'ENVIRONMENT_RECOVERY_DISPATCH.json').write_text(json.dumps(receipt,indent=2)+'\n');assert rc==0,err.decode();print(result.decode())
finally:pw=None;c.close()
