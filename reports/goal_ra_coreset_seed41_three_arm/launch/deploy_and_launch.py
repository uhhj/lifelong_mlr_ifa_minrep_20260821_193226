"""Reviewed source-only upload; credentials in memory, host key verified."""
from pathlib import Path
import json,hashlib,zipfile,getpass,paramiko
R=Path(__file__).resolve().parent;Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
files={}
for arm in ['baseline','ra']:
 for p in (R/arm).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(R).as_posix()]=p
for n in ['PROTOCOL.md','ACTUAL_SPEC.yaml','SOURCE_DIFF.json','source.patch','audit_suite.py','queue.py','supervise_queue.sh','monitor_snapshot.py','aggregate_seed41.py','final_queue_audit.py']:files[n]=R/n
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
for n,p in files.items():
 assert p.stat().st_size<1024**2 and (p.suffix in {'.py','.sh','.md','.json','.yaml','.txt','.sbatch','.patch'} or p.name=='LICENSE')
 if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
(R/'CODE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'reviewed_code.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.writestr(n,p.read_bytes())
 z.writestr('CODE_MANIFEST.json',json.dumps(manifest))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 f=c.open_sftp();f.put(str(archive),'/tmp/ra_goal41_reviewed_code.zip');f.put(str(R/'install_remote.py'),'/tmp/ra_goal41_install.py')
 cmd='/data/mlr_ifa/env/bin/python /tmp/ra_goal41_install.py';_,a,b=c.exec_command(cmd,timeout=180)
 out=a.read().decode();err=b.read().decode();rc=a.channel.recv_exit_status()
 (R/'DEPLOY_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stdout=out,stderr=err),indent=2)+'\n');print(out,err,flush=True);assert rc==0
 for n in ['DEPLOYMENT.json','LAUNCH.json','ORCHESTRATION_SHA256.json']:(R/n).write_bytes(f.open(Q+'/'+n).read())
finally:c.close()
