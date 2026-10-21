"""Fixed read-only SSH snapshot, metadata only; never a resume command."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,time,hashlib,subprocess,shutil
from pathlib import Path
W=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021');O=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):
 if not p.is_file():return None
 return json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-12000:]
files={}
for base in (W,O):
 for n in ('RECOVERY_ENVIRONMENT01.json','RECOVERY_ENVIRONMENT01_FAILURE.json','recovery_environment01.exit','recovery_environment01.log','RUN_STATUS.json','INSTALL_RECEIPT.json','DISPATCH_RECEIPT.json','FAILURE.json','CONTRACT.json','supervisor.pid.json','supervisor.exit','supervisor.log','pipeline.exit','pipeline.pid.json','LAUNCH_ACCEPTANCE.json','GPU_PREFLIGHT.json','TASK6_TENSOR_PARITY.json','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json'):
  if (base/n).is_file():files[str(base/n)]=read(base/n)
for p in (W/'admission').glob('*'):
 if p.suffix in ('.json','.exit','.log'):files[str(p)]=read(p)
for p in O.glob('*.exit'):files[str(p)]=read(p)
for p in O.glob('*.log'):files[str(p)]=read(p)
for pattern in ('effects/current/*/AUDIT.json','effects/current/*/SUPPORT.json','effects/models/*/ACCEPTANCE.json','ra_coreset/functional_references/*.json','ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','preflight/optimizer_calls.json'):
 for p in O.glob(pattern):files[str(p)]=read(p)
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=(p/'cwd').resolve()
  if (cwd==W or cwd==W/'source') and ('caet_v2' in cmd or 'supervisor.py' in cmd or 'recovery_environment01.py' in cmd or 'coreset_exp.evaluate' in cmd or 'spawn_main' in cmd):processes.append(dict(pid=int(p.name),command=cmd))
 except (OSError,PermissionError):pass
epochs=read(O/'ra_coreset/epochs.json') or []
formal=[];attempts=[]
for p in (O/'ra_coreset/evaluation').glob('*.jsonl'):
 rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
 if p.name.startswith('attempts'):attempts.extend(rows)
 else:formal.extend(rows)
auxiliary={};partial_aux=[]
for row in epochs:
 p=O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(row['task'],row['epoch']))
 rows=read(p)
 if rows is None:partial_aux.append([row['task'],row['epoch']]);continue
 summary=auxiliary.setdefault(str(row['task']),dict(epochs=0,updates=0,transfer_applied=0,transfer_blocked=0,retention_active=0,active=0,accounting_pass=True))
 summary['epochs']+=1;summary['updates']+=len(rows)
 summary['accounting_pass'] &= len(rows)==row['batches'] and [x['step'] for x in rows]==list(range(row['steps']-row['batches'],row['steps']))
 summary['transfer_applied']+=sum(x['transfer_applied'] for x in rows)
 summary['transfer_blocked']+=sum(x['transfer_dot'] is not None and x['transfer_dot']<0 for x in rows)
 summary['retention_active']+=sum(x['retention_groups']>0 for x in rows)
 summary['active']+=sum(x['active'] for x in rows)
contract=read(O/'CONTRACT.json');hashes={}
if contract:
 for path,expected in contract['files'].items():hashes[path]=Path(path).is_file() and sha(Path(path))==expected
 for name,expected in contract['source_sha256'].items():
  path=W/'source'/name;hashes['source/'+name]=path.is_file() and sha(path)==expected
print(json.dumps(dict(observed_unix=time.time(),files=files,processes=processes,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),disk={p:shutil.disk_usage(p).free for p in ('/','/data')},auxiliary=auxiliary,partial_aux=partial_aux,epochs=len(epochs),last_epoch=epochs[-1] if epochs else None,formal=formal,attempts=attempts,frozen_hashes=hashes)))
'''

def main():
 p=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n',encoding='utf8');assert rc==0,err.decode()
  d=json.loads(raw);(p/'SNAPSHOT.json').write_bytes(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
  print(json.dumps(dict(directory=str(p),epochs=d['epochs'],last_epoch=d['last_epoch'],formal_records=len(d['formal']),hashes_pass=all(d['frozen_hashes'].values()),processes=d['processes'],status={k:v for k,v in d['files'].items() if k.endswith(('RUN_STATUS.json','GPU_PREFLIGHT.json','FAILURE.json','.exit'))})))
 finally:pw=None;c.close()

if __name__=='__main__':main()
