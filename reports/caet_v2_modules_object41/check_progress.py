"""Fixed read-only snapshot; never launches a model or an environment."""
import datetime,getpass,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,os,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022')
def read(p):
 if not p.exists():return None
 if p.suffix=='.json':return json.loads(p.read_text())
 return p.read_text()[-12000:]
out=dict(unix=time.time(),queue={n:read(Q/n) for n in ('RUN_STATUS.json','FAILURE.json','controller.exit','controller.pid.json','controller.log')},variants={},processes=[],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),disk={})
for mount in ('/','/data'):
 s=os.statvfs(mount);out['disk'][mount]=dict(ordinary_free=s.f_bavail*s.f_frsize,root_free=s.f_bfree*s.f_frsize)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=(p/'cwd').resolve()
  if 'python -c' not in cmd and (str(Q) in cmd or str(cwd).startswith(str(Q))):out['processes'].append(dict(pid=int(p.name),cwd=str(cwd),command=cmd))
 except OSError:pass
contract=read(Q/'ORCHESTRATION_CONTRACT.json');fail=[]
for name,h in contract['files'].items():
 p=Path(name)
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:fail.append(name)
out['orchestration_hashes']=dict(count=len(contract['files']),failures=fail)
for v in ('TAC','RSET','EFTR'):
 w=Q/v;e=read(w/'ENTRY.json');o=Path(e['output']);d=dict(entry=e,worker={n:read(w/n) for n in ('supervisor.pid.json','supervisor.exit','FAILURE.json')},files={},exits={},logs={},receipt=read(Q/(v+'_OBJECT_41.receipt.json')))
 for n in ('RUN_STATUS.json','GPU_PREFLIGHT.json','VARIANT_GPU_PREFLIGHT.json','TASK6_TENSOR_PARITY.json','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','pipeline.exit','supervisor.exit'):
  d['files'][n]=read(o/n)
 for p in (w/'admission').glob('*.exit'):d['exits']['admission/'+p.name]=read(p)
 for p in o.glob('*.exit'):d['exits'][p.name]=read(p)
 for folder in (w/'admission',o):
  logs=sorted(folder.glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]
  for p in logs:d['logs'][str(p)]=read(p)
 h=read(o/'ra_coreset/epochs.json') or [];d['epochs']=len(h);d['last_epoch']=h[-1] if h else None
 d['formal_records']=sum(len(p.read_text().splitlines()) for p in (o/'ra_coreset/evaluation').glob('trials*.jsonl'))
 d['auxiliary']={}
 for t in (7,8,9):
  rows=[x for p in (o/'ra_coreset/auxiliary').glob('task%d_epoch*.json'%t) for x in json.loads(p.read_text())]
  d['auxiliary'][str(t)]=dict(recorded_steps=len(rows),transfer=sum(bool(x['transfer_applied']) for x in rows),retention=sum(x['retention_loss']>0 for x in rows),candidate_accepted=sum(x.get('candidate_accepted',False) for x in rows),candidate_rejected=sum(x.get('candidate_accepted') is False for x in rows),selective_joint=sum((x.get('selective_gate') or {}).get('joint',0) for x in rows))
 d['input_shift']={p.name:read(p) for p in (o/'input_shift').glob('*.json')}
 attempts=[json.loads(line) for p in (o/'ra_coreset/evaluation').glob('attempts*.jsonl') for line in p.read_text().splitlines()]
 d['attempt_events']={key:sum(x['event']==key for x in attempts) for key in sorted({x['event'] for x in attempts})}
 d['effects']={str(t):read(o/'effects/models'/('task%d'%t)/'ACCEPTANCE.json') for t in (7,8,9)}
 out['variants'][v]=d
print(json.dumps(out))
'''
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 p=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n');assert rc==0,err.decode()
  (p/'SNAPSHOT.json').write_bytes(raw);d=json.loads(raw);assert not d['orchestration_hashes']['failures']
  (p/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='READONLY_SNAPSHOT_HASH_CHECK_PASS',formal_acceptance=False,sha256=__import__('hashlib').sha256(raw).hexdigest()),indent=2)+'\n')
  print(json.dumps(dict(directory=str(p),queue=d['queue']['RUN_STATUS.json'],gpu=d['gpu'],variants={v:dict(epochs=x['epochs'],formal=x['formal_records'],status=x['files']['RUN_STATUS.json'],preflight=x['files']['VARIANT_GPU_PREFLIGHT.json'],failure=x['worker']['FAILURE.json']) for v,x in d['variants'].items()}),ensure_ascii=False,indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
