"""Completed entries only: explicit metadata allowlist and manifest, never payload."""
import datetime,getpass,hashlib,json,shlex,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
key=sys.argv[1];assert key in ['OBJECT_42','OBJECT_43','GOAL_41','GOAL_42','GOAL_43']
SCRIPT=r'''
import os,json,hashlib,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022');key='@KEY@'
entry=next(e for e in json.loads((Q/'QUEUE.json').read_text()) if e['key']==key);O=Path(entry['output']);P=Path(entry['parent']);W=Q/key
assert (O/'supervisor.exit').read_text().strip()=='0' and (O/'pipeline.exit').read_text().strip()=='0'
assert json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
assert (Q/(key+'.receipt.json')).exists()
residual=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if (p/'cwd').resolve()==W/'source' and 'python -c' not in cmd:residual.append(int(p.name))
 except OSError:pass
assert not residual,residual
items={}
def add(p,name):
 assert p.suffix in ('.json','.jsonl','.exit') and p.is_file();raw=p.read_bytes();assert len(raw)<8*1024**2
 items[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for p in list(O.glob('*.json'))+list(O.glob('*.exit')):add(p,'current/'+p.name)
patterns=['ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','ra_coreset/TRAINING_COMPLETE.json','ra_coreset/ifa_task*.json','ra_coreset/storage/*.json','ra_coreset/auxiliary/*.json','ra_coreset/constraints/*.json','ra_coreset/evaluation/*.json','ra_coreset/evaluation/*.jsonl','ra_coreset/dataset_manifests/*.json','preflight/optimizer_calls.json','effects/current/*/AUDIT.json','effects/current/*/SUPPORT.json','effects/memory/*.json']
for pattern in patterns:
 for p in O.glob(pattern):add(p,'current/'+str(p.relative_to(O)))
for pattern in ['ra_coreset/INITIAL_STATE.json','ra_coreset/ifa_task*.json','ra_coreset/storage/*.json','ra_coreset/evaluation/trials*.jsonl']:
 for p in P.glob(pattern):add(p,'baseline/'+str(p.relative_to(P)))
for name,path in [('ENTRY.json',W/'ENTRY.json'),('supervisor.exit',W/'supervisor.exit'),('receipt.json',Q/(key+'.receipt.json')),('server_independent.json',Q/'independent'/(key+'.json'))]:add(path,'launcher/'+name)
print(json.dumps(dict(status='LIGHT_MANIFEST_READY',key=key,entry=entry,observed_unix=time.time(),residual_workers=[],entries=items)))
'''.replace('@KEY@',key)
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read().decode();code=o.channel.recv_exit_status();assert code==0,err
 d=R/('final_'+key+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));d.mkdir();(d/'PACKAGE_MANIFEST.json').write_bytes(raw)
 print(json.dumps(dict(directory=str(d),status='MANIFEST_ONLY_RUN_RESUME_LIGHT_DOWNLOAD',files=len(json.loads(raw)['entries']))))
finally:pw=None;c.close()
