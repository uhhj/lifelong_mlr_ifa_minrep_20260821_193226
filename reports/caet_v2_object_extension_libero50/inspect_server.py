import getpass,json,shlex,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,subprocess,shutil,time,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020');out=dict(time=time.time(),disk={x:shutil.disk_usage(x).free for x in ('/','/data')},gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),files={},parents={},processes=[])
for p in Path('/proc').glob('[0-9]*'):
 try:
  c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if any(x in c for x in ('queue_controller','worker_entrypoint','caet_v2_exp.pipeline','l50_exp.')) and 'python -c' not in c:out['processes'].append(dict(pid=int(p.name),command=c))
 except OSError:pass
for n in ('REGISTRY.json','COVR_CANCELLED_BY_USER.json','LIBERO50_SERVER_STATUS.json','LIBERO50_REGISTRY_ACCEPTANCE.json','LIBERO50_CPU_NATIVE.json','LIBERO50_READINESS_20261021.json'):
 p=Q/n
 if p.is_file():out['files'][n]=json.loads(p.read_text())
for seed in (41,43):
 p=Q/'entries'/('CAET_OBJECT_%d.json'%seed);out['files'][p.name]=json.loads(p.read_text())
for p in Path('/data/mlr_ifa/experiments').glob('RA_Coreset_OBJECT_seed*'):
 if not any(('seed%d_'%s) in p.name for s in (41,43)):continue
 out['parents'][str(p)]={}
 for n in ('AUDIT.json','AUTONOMOUS_EXIT_ACCEPTANCE.json','resolved_ra_coreset.json','CONTRACT.json','GPU_PREFLIGHT.json'):
  f=p/n
  if f.exists():out['parents'][str(p)][n]=json.loads(f.read_text())
for p in Path('/data/mlr_ifa/experiments').glob('*LIBERO50*'):out['files'][str(p)]=dict(exists=True,files=[x.name for x in p.iterdir()])
b=Q/'LIBERO50_SOURCE_BUNDLE_final.json';out['l50_bundle_sha256']=hashlib.sha256(b.read_bytes()).hexdigest()
print(json.dumps(out))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(R/'INSPECT_ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode();(R/'SERVER_INSPECT.json').write_bytes(raw)
 # Existing reviewed source text only; no model or dataset payloads.
 s=c.open_sftp()
 for name,remote in [('LIBERO50_SOURCE_BUNDLE.json','/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020/LIBERO50_SOURCE_BUNDLE_final.json'),('CAET_PARENT_SOURCE_BUNDLE.json',None)]:
  if remote:
   with s.open(remote,'rb') as f:(R/name).write_bytes(f.read())
 print('INSPECT_OK')
finally:pw=None;c.close()
