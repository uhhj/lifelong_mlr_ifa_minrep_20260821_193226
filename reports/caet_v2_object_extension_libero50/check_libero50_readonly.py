"""Taskmap01 shared pretraining: frozen identities, GPU gate and real process evidence."""
import datetime,getpass,json,shlex,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,time,hashlib,subprocess,os
from pathlib import Path
D=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021/libero50_taskmap01')
W=D/'shared_pretraining';O=Path('/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021_taskmap01')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
d=dict(files={},file_sha256={},hashes={},processes=[])
for root,names in [(D,['CONTRACT.json','RUN_STATUS.json','controller.pid.json','controller.exit','FAILURE.json','MAPPING_AUDIT.json']), (O,['CONTRACT.json','resolved_pretrain.json','PRETRAIN_GPU_PREFLIGHT.json','LAUNCH_ACCEPTANCE.json','PRETRAIN_INITIALIZATION.json','RUN_STATUS.json','pretraining/epochs.json','PRETRAIN_COMPLETE.json','PRETRAIN_STRICT_LOAD.json','PROCESS_EXIT_ACCEPTANCE.json','supervisor.exit','pipeline.exit','pretrain.exit','pretrain.log']), (W/'admission',['software.exit','ra_software.exit','prepare.exit','gpu_preflight.exit','pipeline.exit','final_process.exit'])]:
 for n in names:
  p=root/n
  if p.is_file():
   d['file_sha256'][str(p)]=sha(p)
   d['files'][str(p)]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-12000:]
c=json.loads((O/'CONTRACT.json').read_text())
for p,h in c['files'].items():d['hashes'][p]=sha(p)==h
for n,h in c['source_hashes'].items():d['hashes']['source/'+n]=sha(W/'source'/n)==h
for p,h in json.loads((D/'CONTRACT.json').read_text())['files'].items():d['hashes']['dispatch/'+p]=sha(p)==h
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'python -c' not in cmd and any(x in cmd for x in ('libero50_taskmap01','l50_exp.')):d['processes'].append(dict(pid=int(p.name),command=cmd))
 except OSError:pass
d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
d['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
d['disk']={}
for m in ('/','/data'):
 st=os.statvfs(m);d['disk'][m]=dict(available=st.f_bavail*st.f_frsize,free_all=st.f_bfree*st.f_frsize)
d['observed_unix']=time.time();print(json.dumps(d))
'''
dest=R/('libero50_progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);i.channel.shutdown_write();raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
 (dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode()
 json.loads(raw);(dest/'SNAPSHOT.json').write_bytes(raw);(dest/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n');print(dest)
finally:pw=None;c.close()
