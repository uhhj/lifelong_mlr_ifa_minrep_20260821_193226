"""Create one new finite diagnostic root. Never overwrite or restart."""
import ast,datetime,getpass,hashlib,json,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1'
names=['paired_worker.py','supervisor.py','INPUT_SPEC.json','PAIRED_PROTOCOL.md']
for n in names:
 if n.endswith('.py'):ast.parse((R/n).read_text())
contract={'version':'RA_ALIGN_PAIRED_GOAL_v1','files':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in names},'private_root':'/root/mlr_ifa/experiments/RA_ALIGN_PAIRED_GOAL_20261024_v1','episodes':240,'new_training_updates':0}
(R/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
attempt=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));attempt.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 code="""import os,json,subprocess
from pathlib import Path
q=Path(%r)
assert not q.exists(), 'Root exists; do not redeploy'
assert not Path(%r).exists(), 'Private root exists'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not gpu, gpu
s=os.statvfs('/root');assert s.f_bavail*s.f_frsize>=4*1024**3
s=os.statvfs('/data');assert s.f_bfree*s.f_frsize>=2*1024**3
q.mkdir(parents=True)
print(json.dumps({'preflight':'PASS_GPU_IDLE_CAPACITY','root':str(q)}))
"""%(Q,contract['private_root'])
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -');i.write(code);i.flush();i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();(attempt/'PREFLIGHT.json').write_text(json.dumps({'exit':rc,'stdout':out,'stderr':err},indent=2));assert rc==0,err
 sf=c.open_sftp()
 for n in names+['CONTRACT.json']:sf.put(str(R/n),Q+'/'+n)
 sf.close()
 code="""import subprocess,json
from pathlib import Path
q=Path(%r)
with (q/'supervisor.log').open('xb') as f:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(q/'supervisor.py')],cwd=q,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'root':str(q),'pid':p.pid}))
"""%Q
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -');i.write(code);i.flush();i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
 d=json.loads(out);d.update(contract_sha256=hashlib.sha256((R/'CONTRACT.json').read_bytes()).hexdigest(),time=datetime.datetime.now().astimezone().isoformat(),status='LAUNCHED_ADMISSION_PENDING')
 (R/'ACTIVE_ROOT.json').write_text(json.dumps(d,indent=2));(attempt/'LAUNCH.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
finally:pw=None;c.close()
