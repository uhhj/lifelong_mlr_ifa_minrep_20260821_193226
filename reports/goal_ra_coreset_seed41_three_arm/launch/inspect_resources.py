"""Read-only inventory for requested GOAL seed41; no model or environment calls."""
from pathlib import Path
import json,getpass,datetime,paramiko
R=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,shutil,subprocess,time
roots=['/root/mlr_ifa/experiments','/data/mlr_ifa/experiments','/root/mlr_ifa_new_experiments','/root/mlr_ifa/experiment_workspaces','/data/mlr_ifa/experiment_workspaces']
matches=[]
for root in roots:
 p=Path(root)
 if not p.exists():continue
 for q in p.iterdir():
  if not q.is_dir():continue
  if '41' in q.name or 'GOAL' in q.name or 'goal' in q.name:
   children=[x.name for x in q.iterdir() if x.is_dir() and ('41' in x.name or x.name in ['random','coreset','ra_coreset','features'])]
   matches.append(dict(path=str(q),children=children,status=json.loads((q/'RUN_STATUS.json').read_text()) if (q/'RUN_STATUS.json').exists() else None))
out=dict(time=time.time(),matches=matches,free_bytes={x:shutil.disk_usage(x).free for x in ['/root','/data']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),scientific_processes=[x for x in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines() if ' -m coreset_exp.' in x],new_training_or_rollouts=0)
print(json.dumps(out,indent=2))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 command="/data/mlr_ifa/env/bin/python - <<'PY'\n"+code+"\nPY"
 _,o,e=c.exec_command(command,timeout=120);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 (R/'RESOURCE_COMMAND.json').write_text(json.dumps(dict(command=command,exit_code=rc,stderr=err,local_time=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 assert rc==0,err
 a=json.loads(raw);(R/'RESOURCE_INVENTORY.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
finally:c.close()
