"""Versioned AUX8 dispatcher; old TAC remains stopped, original B/C stay immutable."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_AUX8_OBJECT41_20261022'
SCRIPT=r'''
import sys,json,hashlib,subprocess,shutil,time,os,fcntl
from pathlib import Path
d=json.load(sys.stdin);Q=Path(d['q']);assert not Q.exists()
OLD=Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022');stop=json.loads((OLD/'TAC_STOPPED_FOR_AUX8_USER_REVISION.json').read_text());assert not stop['remaining']
lock=(OLD/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd:assert not (str(OLD) in cmd or cwd.startswith(str(OLD))),('old process remains',p.name,cmd)
 except (OSError,ValueError):pass
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldcontract=json.loads((OLD/'ORCHESTRATION_CONTRACT.json').read_text());assert all(sha(p)==h for p,h in oldcontract['files'].items())
BASE=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021/source');expected=json.loads(Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021/CONTRACT.json').read_text())['source_sha256'];assert all(sha(BASE/n)==h for n,h in expected.items())
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019');assert json.loads((P/'AUTONOMOUS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
Q.mkdir();W=Q/'TAC_AUX8';W.mkdir();S=W/'source';shutil.copytree(BASE,S,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files={}
def wr(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for n,s in d['source'].items():
 compile(s,n,'exec');(S/n).write_text(s)
for n,s in d['top'].items():(Q/n).write_text(s);files[str(Q/n)]=sha(Q/n)
for n,s in d['worker'].items():(W/n).write_text(s);files[str(W/n)]=sha(W/n)
out='/data/mlr_ifa/experiments/CAETv2_TAC_AUX8_OBJECT_seed41_20261022';assert not Path(out).exists()
wr(W/'ENTRY.json',dict(seed=41,variant='TAC_AUX8',parent=str(P),output=out));files[str(W/'ENTRY.json')]=sha(W/'ENTRY.json')
actual={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml') and '__pycache__' not in p.parts}
changes={k:dict(before=expected.get(k),after=h) for k,h in actual.items() if expected.get(k)!=h};assert all(k.startswith('caet_v2_exp/') for k in changes)
files.update({str(S/k):h for k,h in actual.items()})
for p,h in oldcontract['files'].items():
 if p.startswith(str(OLD/'RSET')+'/') or p.startswith(str(OLD/'EFTR')+'/') or p==str(OLD/'seed_supervisor.py'):files[p]=h
oldqueue=json.loads((OLD/'QUEUE.json').read_text());pending=oldqueue[1:];assert [x['key'] for x in pending]==['RSET_OBJECT_41','EFTR_OBJECT_41']
for e in pending:assert not Path(e['output']).exists()
entries=[dict(key='TAC_AUX8_OBJECT_41',method='CAET-v2+TAC_AUX8',seed=41,output=out,storage_mount='/data',minimum_free_bytes=1600000000,argv=['/data/mlr_ifa/env/bin/python','-u',str(Q/'seed_supervisor.py'),str(W)])]+pending
wr(Q/'QUEUE.json',entries);files[str(Q/'QUEUE.json')]=sha(Q/'QUEUE.json')
wr(Q/'SOURCE_DIFF.json',dict(files=len(actual),unchanged=sum(expected.get(k)==h for k,h in actual.items()),changes=changes))
wr(Q/'ORCHESTRATION_CONTRACT.json',dict(files=files,created_unix=time.time(),old_stop_sha256=sha(OLD/'TAC_STOPPED_FOR_AUX8_USER_REVISION.json')))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as log:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'queue_controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='DISPATCHED_AWAITING_OWN_GPU_ADMISSION',controller_pid=p.pid,queue=entries,source_count=len(actual),source_unchanged=sum(expected.get(k)==h for k,h in actual.items()),changes=changes,contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),old_contract_verified=True,old_processes_absent=True,unix=time.time())
wr(Q/'DISPATCH_RECEIPT.json',r);print(json.dumps(r))
'''
def main():
 gate=(R/'GATE1_EVIDENCE.json').read_text(encoding='utf8');audit=(R/'GATE1_LOCAL_AUDIT.json').read_text(encoding='utf8');assert json.loads(audit)['status']=='GATE1_PASS';assert hashlib.sha256(gate.encode()).hexdigest()==json.loads(audit)['evidence_sha256']
 top={n:(R/n).read_text(encoding='utf8') for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md','queue_controller.py','seed_supervisor.py')}
 data=dict(q=Q,source={p.relative_to(R/'additions').as_posix():p.read_text(encoding='utf8') for p in (R/'additions').rglob('*.py')},top=top,worker={n:top[n] for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md')})
 data['worker'].update({'GATE1_EVIDENCE.json':gate,'GATE1_LOCAL_AUDIT.json':audit})
 p=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=240);i.write(json.dumps(data));i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (p/'DISPATCH_RECEIPT.json').write_bytes(raw);r=json.loads(raw);print(json.dumps({k:r[k] for k in ('status','controller_pid','source_count','source_unchanged','old_processes_absent')},indent=2));print(p)
 finally:pw=None;c.close()
if __name__=='__main__':main()
