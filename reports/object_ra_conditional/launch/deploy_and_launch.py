"""Upload reviewed source-only archive, install and launch authorized finite queue once."""
from pathlib import Path
import getpass,json,hashlib,zipfile
import paramiko
R=Path(__file__).resolve().parent;Q='/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019'
files={}
for seed in (41,42,43):
 for p in (R/'source').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files['seed%d/source/'%seed+p.relative_to(R/'source').as_posix()]=p
 for p in (R/('seed%d'%seed)).iterdir():
  if p.is_file():files['seed%d/'%seed+p.name]=p
for n in ['PROTOCOL.md','DECISION_GRAPH.json','PARENTS.json','SOURCE_DIFF.json','source.patch','queue.py','supervise_queue.sh','monitor_snapshot.py','aggregate_stage1.py','final_queue_audit.py']:files[n]=R/n
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
for n,p in files.items():
 assert p.stat().st_size<1024**2 and (p.suffix in {'.py','.sh','.md','.json','.yaml','.txt','.sbatch','.patch'} or p.name=='LICENSE')
(R/'CODE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'reviewed_code.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.writestr(n,p.read_bytes())
 z.writestr('CODE_MANIFEST.json',json.dumps(manifest))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 f=c.open_sftp();f.put(str(archive),'/tmp/ra_object_conditional_20261019_reviewed_code.zip');f.put(str(R/'install_remote.py'),'/tmp/ra_object_conditional_20261019_install.py')
 cmd='/data/mlr_ifa/env/bin/python /tmp/ra_object_conditional_20261019_install.py';_,a,b=c.exec_command(cmd,timeout=120)
 out=a.read().decode();err=b.read().decode();rc=a.channel.recv_exit_status()
 (R/'DEPLOY_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stdout=out,stderr=err),indent=2)+'\n');print(out,err,flush=True);assert rc==0
 for n in ['DEPLOYMENT.json','LAUNCH.json','ORCHESTRATION_SHA256.json']:(R/n).write_bytes(f.open(Q+'/'+n).read())
finally:c.close()
