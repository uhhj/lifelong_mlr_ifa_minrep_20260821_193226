"""Explicit user cancellation; preserve science, outputs and original exit files."""
import getpass,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
CODE=r'''
import json,os,signal,time,hashlib
from pathlib import Path
q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
o=Path('/data/mlr_ifa/experiments/COVR_OBJECT_seed42_full_20261021')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def procs():
 r={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   stat=(p/'stat').read_text().split(') ',1)[1].split()
   r[int(p.name)]=dict(pid=int(p.name),ppid=int(stat[1]),state=stat[0],start=stat[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
  except OSError:pass
 return r
pid=read(q/'controller_v3.pid.json')['pid'];ps=procs()
assert pid in ps and 'queue_controller_v3.py' in ps[pid]['command']
assert read(q/'DECISION.json')['method']=='COVR'
os.kill(pid,signal.SIGSTOP)
ps=procs();targets={pid}
while True:
 new={p for p,r in ps.items() if r['ppid'] in targets}
 if new<=targets:break
 targets|=new
assert any('covr_exp.' in ps[p]['command'] for p in targets)
assert not any('libero50' in ps[p]['command'].lower() for p in targets)
e=dict(status='STOPPING_BY_USER',unix=time.time(),reason='User explicitly cancelled all COVR tasks; first analyze completed CAET OBJECT results.',
 cancelled_method='COVR',cancelled_suites=['OBJECT','GOAL'],cancelled_seeds=[41,42,43],
 processes_before=[ps[p] for p in sorted(targets)],run_status=read(o/'RUN_STATUS.json'),
 controller_dispatch_stopped=True,libero50_cancelled=False,original_exits_preserved=True,
 contract_sha256=hashlib.sha256((o/'CONTRACT.json').read_bytes()).hexdigest())
path=q/'COVR_CANCELLED_BY_USER.json';assert not path.exists()
path.write_text(json.dumps(e,indent=2)+'\n')
for p in targets-{pid}:
 try:os.kill(p,signal.SIGSTOP)
 except ProcessLookupError:pass
for p in targets:
 try:os.kill(p,signal.SIGTERM)
 except ProcessLookupError:pass
for p in targets:
 try:os.kill(p,signal.SIGCONT)
 except ProcessLookupError:pass
time.sleep(3)
for p,r in procs().items():
 if p in targets and r['start']==ps[p]['start'] and r['state']!='Z':os.kill(p,signal.SIGKILL)
time.sleep(1)
remaining=[r for p,r in procs().items() if p in targets and r['start']==ps[p]['start'] and r['state']!='Z']
assert not remaining,remaining
e.update(status='STOPPED_BY_USER',completed_unix=time.time(),remaining_target_processes=remaining,
 epochs=len(read(o/'ra_coreset/epochs.json') or []),new_scientific_acceptance=False,
 collections={p.name:dict(committed=len(list((p/'committed').glob('*.json'))),acceptance=read(p/'ACCEPTANCE.json')) for p in (o/'covr').glob('stage*')})
path.write_text(json.dumps(e,indent=2)+'\n');print(json.dumps(e))
'''
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
 pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  _,out,err=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(CODE),timeout=60)
  raw=out.read();error=err.read();rc=out.channel.recv_exit_status();assert rc==0,(rc,error.decode())
  d=json.loads(raw);(R/'COVR_CANCELLED_BY_USER.json').write_bytes(raw)
  print(json.dumps({k:d[k] for k in ['status','epochs','remaining_target_processes','libero50_cancelled']},indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
