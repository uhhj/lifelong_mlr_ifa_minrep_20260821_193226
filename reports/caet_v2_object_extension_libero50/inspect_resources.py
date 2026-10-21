import getpass,json,shlex,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,shutil,os,time,h5py
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020');d=json.loads((Q/'LIBERO50_PARTITION.json').read_text());out={'disk':{p:shutil.disk_usage(p).free for p in ('/','/data')},'windows':[],'sizes':{},'launchers':{},'memory':Path('/proc/meminfo').read_text()}
for task in sum(d['lifelong_stages'],[]):
 with h5py.File(Path('/data/mlr_ifa/datasets/libero50_acquired')/task['dataset_relative_path'],'r') as h:
  demos=sorted(h['data'],key=lambda x:int(x.split('_')[-1]))[:10];out['windows'].append(sum(h['data'][n]['actions'].shape[0] for n in demos))
P=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021')
for p in P.rglob('*.pth'):out['sizes'][str(p.relative_to(P))]=p.stat().st_size
out['feature_bytes']=next((P/'features/6').glob('*.pt')).stat().st_size
for n in ('RA_IFA_LIBERO50_41','RA_IFA_LIBERO50_42','RA_IFA_LIBERO50_43'):
 out['launchers'][n]=(Q/'entries'/(n+'.sh')).read_text()
print(json.dumps(out))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(R/'RESOURCE_ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode();(R/'RESOURCE_INSPECT.json').write_bytes(raw)
 print('INSPECT_OK')
finally:pw=None;c.close()
