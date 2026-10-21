"""Install a new, separately contracted capacity-only watcher. No active worker changes."""
import datetime,getpass,json,hashlib,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021'
SCRIPT=r'''import json,hashlib,subprocess,time,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021');D=Q/'capacity_recovery_v1';p=Q/'capacity_recovery_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists();compile(p.read_text(),str(p),'exec')
original=Q/'ORCHESTRATION_CONTRACT.json';before=sha(original)
for f,h in json.loads(original.read_text())['files'].items():assert sha(f)==h,f
assert not (Q/'controller.exit').exists()
pid=json.loads((Q/'controller.pid.json').read_text())['pid']
assert str(Q/'queue_controller.py') in Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
st=os.statvfs('/data');assert os.geteuid()==0 and st.f_bfree>st.f_bavail
D.mkdir()
c=dict(version='capacity_recovery_v1',created_unix=time.time(),original_controller_pid=pid,original_orchestration_sha256=before,recovery_script_sha256=sha(p),mode='Armed; only after original resource guard fails and releases queue lock',allowed=['RA_IFA_LIBERO50_42','RA_IFA_LIBERO50_43'],data_emergency_reserve_bytes=2*1024**3,filesystem_settings_changed=False,original_contract_changed=False,worker_source_changed=False,files_deleted=False)
(D/'CONTRACT.json').write_text(json.dumps(c,indent=2)+'\n')
with (D/'watcher.log').open('x') as log:
 proc=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(p)],cwd=Q,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
assert sha(original)==before
print(json.dumps(dict(status='CAPACITY_WATCHER_DISPATCHED',pid=proc.pid,contract=c,capacity=dict(data_free_all=st.f_bfree*st.f_frsize,data_available_unprivileged=st.f_bavail*st.f_frsize),active_experiment_interrupted=False)))'''
assert json.loads((R/'CAPACITY_POLICY_TEST.json').read_text())['status']=='CAPACITY_POLICY_BOUNDARIES_PASS'
dest=R/('capacity_deploy_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 s=c.open_sftp();remote=Q+'/capacity_recovery_v1.py'
 try:s.stat(remote)
 except FileNotFoundError:pass
 else:raise RuntimeError('Recovery already exists; do not overwrite')
 with s.open(remote,'wb') as f:f.write((R/'capacity_recovery_v1.py').read_bytes())
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
 (dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n',encoding='utf8');assert rc==0,err.decode()
 (dest/'RECEIPT.json').write_bytes(raw);print(raw.decode());print(dest)
finally:pw=None;c.close()
