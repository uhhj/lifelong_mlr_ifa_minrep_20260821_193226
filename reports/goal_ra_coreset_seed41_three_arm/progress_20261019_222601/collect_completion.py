"""Collect verified metadata-only packages after successful queue exit."""
from pathlib import Path
import json, hashlib, getpass, paramiko, sys, time
R=Path(__file__).resolve().parent; D=R/sys.argv[1]
Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
roots={'baseline':'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019','ra':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019'}
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
commands=[]; summaries={}
try:
 f=c.open_sftp()
 for n in ['queue.exit','wrapper.exit']:
  assert f.open(Q+'/'+n).read().strip()==b'0'
 for name,root in roots.items():
  cmd='CORESET_OUT='+root+' /data/mlr_ifa/env/bin/python '+Q+'/ra/package_light.py'
  _,a,b=c.exec_command(cmd,timeout=120);out=a.read().decode();err=b.read().decode();rc=a.channel.recv_exit_status()
  commands.append(dict(command=cmd,exit_code=rc,stderr=err));assert rc==0,err
  remote=out.strip();assert remote.startswith(Q+'/ra/light_')
  raw=f.open(remote+'/PACKAGE_MANIFEST.json').read();m=json.loads(raw)
  dest=D/('completion_'+name);dest.mkdir(exist_ok=True)
  for rel,v in m['files'].items():
   p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
   assert not {'bindings','features','buffer','models','recovery','smoke'} & set(p.parts)
   assert p.suffix in {'.json','.jsonl','.log','.exit','.pid','.started'}
   assert v['bytes']<8*1024**2
   data=f.open(remote+'/'+rel).read();assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
   target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  (dest/'PACKAGE_MANIFEST.json').write_bytes(raw)
  summaries[name]=dict(remote=remote,files=len(m['files']),manifest_sha256=hashlib.sha256(raw).hexdigest())
  time.sleep(1.1)
 (D/'PACKAGE_DOWNLOAD_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',commands=commands,packages=summaries,weights_or_payloads_downloaded=False),indent=2)+'\n')
 print(json.dumps(summaries,indent=2))
finally:
 c.close()
