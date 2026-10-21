"""Fixed-scope metadata check. No policy imports, server writes or environment calls."""
import argparse,datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
CODE=r'''
import json,time,hashlib,subprocess
from pathlib import Path
q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
e=read(q/'REGISTRY.json')['COVR_OBJECT_42'];o=Path(e['output']);s=Path(e['source'])
r=dict(unix=time.time(),decision=read(q/'DECISION.json'),queue_status=read(q/'RUN_STATUS.json'),
 controller_exit=(q/'controller_v3.exit').read_text() if (q/'controller_v3.exit').exists() else None,
 blocked=read(q/'BLOCKED.json'),output_exists=o.exists(),server_writes=0,optimizer_calls=0,environment_steps=0)
if o.exists():
 c=read(o/'CONTRACT.json');assert c and c['seed']==42
 for p,h in c['files'].items():assert sha(p)==h,p
 for p,h in c['source_sha256'].items():assert sha(s/p)==h,p
 r['contract']=dict(sha256=sha(o/'CONTRACT.json'),sources=len(c['source_sha256']),inputs=len(c['files']),hashes='PASS')
 r['preflight']=read(o/'GPU_PREFLIGHT.json');r['run_status']=read(o/'RUN_STATUS.json')
 r['epochs']=read(o/'ra_coreset/epochs.json') or []
 r['exits']={p.name:p.read_text().strip() for p in o.glob('*.exit')}
 r['phase_events']=[json.loads(x) for x in (o/'commands.jsonl').read_text().splitlines()] if (o/'commands.jsonl').exists() else []
 root=o/'ra_coreset';parent=Path(e['baseline'])/'ra_coreset'
 r['task6_tensor']=read(o/'TASK6_TENSOR_PARITY.json')
 r['initialization_equal']=read(root/'INITIAL_STATE.json')==read(parent/'INITIAL_STATE.json')
 r['ifa_equal']={p.name:read(p)==read(parent/p.name) for p in root.glob('ifa_task*.json')}
 r['storage']={}
 for p in sorted((root/'storage').glob('task*.json')):
  if not (o/('retain'+p.stem[4:]+'.exit')).exists() and not (o/'train6.exit').exists():continue
  m=read(p);b=read(parent/'storage'/p.name)
  assert m['K']==b['K'] and m['selection']['selected_ids']==b['selection']['selected_ids']
  identity=lambda z:[(Path(x['source']).name,x['destination'],x['sha256']) for x in z['selected']]
  assert identity(m)==identity(b)
  assert all(sha(x['source'])==sha(root/'buffer'/x['destination'])==x['sha256'] for x in m['selected'])
  r['storage'][p.name]=dict(count=m['K'],ordered_ids_equal=True,payloads_equal=True)
 r['collections']={}
 for stage in sorted((o/'covr').glob('stage*')):
  def lines(p):
   raw=p.read_bytes() if p.exists() else b''
   return [json.loads(x) for x in raw.splitlines(keepends=True) if x.endswith(b'\n')]
  branches=[read(p) for p in sorted((stage/'committed').glob('*.json'))]
  for x in branches:assert sha(stage/'private'/(x['key']+'.pth'))==x['payload_sha256']
  r['collections'][stage.name]=dict(branches=branches,events=lines(stage/'attempts.jsonl'),
    normal=[read(p) for p in sorted((stage/'normal').glob('*.json'))],decisions=read(stage/'DECISIONS.json') or {},
    gates=[read(p) for p in sorted(stage.glob('gate_*.json'))],acceptance=read(stage/'ACCEPTANCE.json'),payload_hashes='PASS')
 r['auxiliary']={}
 for row in r['epochs']:
  p=root/'auxiliary'/('task%d_epoch%02d.json'%(row['task'],row['epoch']))
  stats=read(p);assert stats is not None and len(stats)==row['batches']
  key=str(row['task']);v=r['auxiliary'].setdefault(key,dict(updates=0,active=0))
  v['updates']+=len(stats);v['active']+=sum(x['auxiliary'] is not None for x in stats)
r['processes']=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines()
r['processes']=[x for x in r['processes'] if ('covr_exp.' in x or 'controller_v3.py' in x or 'COVR_OBJECT_42.env_v2.sh' in x) and 'python -c' not in x]
r['observed_end_unix']=time.time();print(json.dumps(r))
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args()
 assert a.directory.resolve().parent==R and (a.directory/'SNAPSHOT.json').exists()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
 pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  _,out,err=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(CODE),timeout=180)
  raw=out.read();error=err.read();rc=out.channel.recv_exit_status();assert rc==0,(rc,error.decode())
  d=json.loads(raw);(a.directory/'COVR_DETAILS.json').write_bytes(raw)
  (a.directory/'COVR_DETAILS_RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),exit=rc,server_writes=0),indent=2)+'\n')
  print(json.dumps(dict(directory=str(a.directory),phase=d.get('run_status'),epochs=len(d.get('epochs',[])),collections={k:len(v['branches']) for k,v in d.get('collections',{}).items()},hashes=d.get('contract')),indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
