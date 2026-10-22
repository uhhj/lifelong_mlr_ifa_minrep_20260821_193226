"""New three-entry allowlist; never resumes any user-cancelled run."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022'
SCRIPT=r'''
import sys,json,hashlib,subprocess,shutil,time,os,fcntl
from pathlib import Path
d=json.load(sys.stdin);Q=Path(d['q']);assert not Q.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd:assert '/experiment_workspaces/' not in cmd and '/experiment_workspaces/' not in cwd,('existing worker',p.name)
 except (OSError,ValueError):pass
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021/source');expected=json.loads(Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021/CONTRACT.json').read_text())['source_sha256'];assert all(sha(BASE/n)==h for n,h in expected.items())
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019');assert json.loads((P/'AUTONOMOUS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
v=os.statvfs('/data');assert v.f_bfree*v.f_frsize>8*1024**3
Q.mkdir();files={};entries=[]
def wr(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for n,s in d['top'].items():(Q/n).write_text(s);files[str(Q/n)]=sha(Q/n)
for variant in ('TAC','EFTR','ALIGN'):
 W=Q/variant;W.mkdir();S=W/'source';shutil.copytree(BASE,S,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n,s in d['source'].items():
  compile(s,n,'exec');p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
 out='/data/mlr_ifa/experiments/CAETv3_%s_OBJECT_seed41_20261022'%variant;assert not Path(out).exists()
 wr(W/'ENTRY.json',dict(seed=41,variant=variant,parent=str(P),output=out));files[str(W/'ENTRY.json')]=sha(W/'ENTRY.json')
 for n,s in d['worker'].items():(W/n).write_text(s);files[str(W/n)]=sha(W/n)
 actual={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml') and '__pycache__' not in p.parts}
 assert all(actual[k]==h for k,h in expected.items());assert all(k in expected or k.startswith('caet_v3_exp/') for k in actual)
 files.update({str(S/k):h for k,h in actual.items()})
 entries.append(dict(key=variant+'_OBJECT_41',method={'TAC':'CAET-v3-TAC','EFTR':'RA-EFTR-v3','ALIGN':'RA-TAC-Align'}[variant],seed=41,output=out,storage_mount='/data',minimum_free_bytes=1600000000,argv=['/data/mlr_ifa/env/bin/python','-u',str(Q/'seed_supervisor.py'),str(W)]))
wr(Q/'QUEUE.json',entries);files[str(Q/'QUEUE.json')]=sha(Q/'QUEUE.json')
wr(Q/'ORCHESTRATION_CONTRACT.json',dict(files=files,created_unix=time.time(),all_original_source_unchanged=True))
with (Q/'controller.log').open('x') as log:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'queue_controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='DISPATCHED_AWAITING_OWN_GPU_ADMISSION',controller_pid=p.pid,queue=entries,contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),frozen_files=len(files),all_original_source_unchanged=True,unix=time.time())
wr(Q/'DISPATCH_RECEIPT.json',r);print(json.dumps(r))
'''
def main():
 gate=(R/'GATE1_EVIDENCE.json').read_text(encoding='utf8');audit=(R/'GATE1_LOCAL_AUDIT.json').read_text(encoding='utf8');assert json.loads(audit)['status']=='PASS';assert hashlib.sha256(gate.encode()).hexdigest()==json.loads(audit)['evidence_sha256']
 top={n:(R/n).read_text(encoding='utf8') for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md','queue_controller.py','seed_supervisor.py')}
 data=dict(q=Q,source={p.relative_to(R/'additions').as_posix():p.read_text(encoding='utf8') for p in (R/'additions').rglob('*.py')},top=top,worker={n:top[n] for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md')})
 data['worker'].update({'GATE1_EVIDENCE.json':gate,'GATE1_LOCAL_AUDIT.json':audit})
 out=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=240);i.write(json.dumps(data));i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (out/'DISPATCH_RECEIPT.json').write_bytes(raw);print(out);print(raw.decode())
 finally:pw=None;c.close()
if __name__=='__main__':main()
