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
 r['initialization']=read(o/'INITIALIZATION_AUDIT.json')
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
  print(json.dumps({k:v for k,v in d.items() if k not in ('epochs','phase_events')},indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
