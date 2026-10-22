"""Read-only new dispatcher snapshot. Never starts or resumes any experiment."""
import datetime,getpass,json,shlex,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import os,time,json,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
def read(p):
 if not p.exists():return None
 if p.suffix=='.json':return json.loads(p.read_text())
 return p.read_text()[-10000:]
entries=read(Q/'QUEUE.json');out=dict(unix=time.time(),queue={n:read(Q/n) for n in ('RUN_STATUS.json','controller.pid.json','controller.exit','FAILURE.json','controller.log')},runs={},processes=[],disk={},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd and ('CAETv3_' in cmd or 'CAETv3_' in cwd):out['processes'].append(dict(pid=int(p.name),command=cmd,cwd=cwd))
 except OSError:pass
c=read(Q/'ORCHESTRATION_CONTRACT.json');out['hashes']=dict(count=len(c['files']),failures=[p for p,h in c['files'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h])
for m in ('/','/data'):
 s=os.statvfs(m);out['disk'][m]=dict(ordinary=s.f_bavail*s.f_frsize,root=s.f_bfree*s.f_frsize)
for entry in entries:
 w=Path(entry['argv'][-1]);o=Path(entry['output']);d=dict(entry=entry,worker={n:read(w/n) for n in ('supervisor.pid.json','supervisor.exit','FAILURE.json')},files={},exits={},logs={})
 for n in ('RUN_STATUS.json','GPU_PREFLIGHT.json','TASK6_TENSOR_PARITY.json','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','pipeline.exit','supervisor.exit','ra_coreset/INITIAL_STATE.json','preflight/optimizer_calls.json'):
  d['files'][n]=read(o/n)
 for folder in (w/'admission',o):
  for p in folder.glob('*.exit'):d['exits'][str(p)]=read(p)
  for p in sorted(folder.glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]:d['logs'][str(p)]=read(p)
 contract=read(o/'CONTRACT.json')
 d['inputs']=None if contract is None else dict(count=len(contract['files']),failures=[p for p,h in contract['files'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h])
 d['parent_initial']=read(Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')/'ra_coreset/INITIAL_STATE.json')
 history=read(o/'ra_coreset/epochs.json') or [];d['epochs']=history;d['auxiliary']={}
 for t in (6,7,8,9):
  stats=[x for p in sorted((o/'ra_coreset/auxiliary').glob('task%d_epoch*.json'%t)) for x in read(p)]
  d['auxiliary'][str(t)]=dict(records=len(stats),head_updates=sum(x.get('head_updates',0) for x in stats),effect_exposures=sum(x.get('effect_samples',0) for x in stats),cross_anchors=sum(x.get('cross_anchors',0) for x in stats),partner_forwards=sum(x.get('extra_partner_forwards',0) for x in stats),optimizer_trials=sum(x.get('optimizer_trials',0) for x in stats),candidate_selected={str(i):sum(x.get('selected')==i for x in stats) for i in range(3)},budget_unmet=sum(x.get('budget_unmet',False) for x in stats))
 d['formal_records']=sum(len(p.read_text().splitlines()) for p in (o/'ra_coreset/evaluation').glob('trials*.jsonl'));d['receipt']=read(Q/(entry['key']+'.receipt.json'))
 out['runs'][entry['key']]=d
print(json.dumps(out))
'''
def main():
 p=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (p/'SNAPSHOT.json').write_bytes(raw);d=json.loads(raw);assert not d['hashes']['failures'];(p/'LOCAL_HASH_AUDIT.json').write_text(json.dumps(dict(status='PASS',count=d['hashes']['count'],snapshot_sha256=hashlib.sha256(raw).hexdigest(),formal_acceptance=False),indent=2))
  print(json.dumps(dict(directory=str(p),queue=d['queue'],gpu=d['gpu'],runs={k:dict(epochs=len(v['epochs']),last=v['epochs'][-1] if v['epochs'] else None,status=v['files']['RUN_STATUS.json'],gate=v['files']['GPU_PREFLIGHT.json'],failure=v['worker']['FAILURE.json'],logs=v['logs']) for k,v in d['runs'].items()}),ensure_ascii=False,indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
