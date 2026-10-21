"""Reviewed explicit five-entry launch; no previous controller is restarted."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021'
SCRIPT=r'''import json,hashlib,shutil,subprocess,time,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021')
OLD=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
BASE=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021/source')
PY='/data/mlr_ifa/env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
assert (OLD/'COVR_CANCELLED_BY_USER.json').is_file()
assert not (Q/'controller.pid.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'python -c' in cmd:continue
  assert not any(x in cmd for x in ('queue_controller_v3.py','queue_controller_v2.py','worker_entrypoint_libero50.py','caet_v2_exp.pipeline','covr_exp.pipeline')),cmd
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
expected=json.loads(Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021/CONTRACT.json').read_text())['source_sha256']
assert expected and all(sha(BASE/n)==h for n,h in expected.items())
files={};entries=[];proof={}
for seed in (41,43):
 w=Q/('object_seed%d'%seed);s=w/'source';assert not s.exists()
 shutil.copytree(BASE,s,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 assert all(sha(s/n)==h for n,h in expected.items())
 proof[str(seed)]=dict(source_files=len(expected),source_exact_seed42=True,hashes=expected)
 out='/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed%d_20261021'%seed;assert not Path(out).exists()
 parent='/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed
 assert json.loads(Path(parent,'AUTONOMOUS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
 entry=dict(seed=seed,parent=parent,output=out)
 wr(w/'ENTRY.json',entry)
 for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md'):(w/n).write_bytes((Q/n).read_bytes())
 for p in (w/'ENTRY.json',w/'PROTOCOL.md',w/'GATE1_EVIDENCE.json',w/'GATE1_LOCAL_AUDIT.json'):files[str(p)]=sha(p)
 entries.append(dict(key='CAETv2_OBJECT_%d'%seed,method='CAETv2_RA_IFA',seed=seed,output=out,storage_mount='/data',minimum_free_bytes=1600000000,argv=[PY,'-u',str(Q/'seed_supervisor.py'),str(w)]))
for seed in (41,42,43):
 e=json.loads((OLD/'entries'/('RA_IFA_LIBERO50_%d.json'%seed)).read_text())
 assert e['method']=='RA_IFA' and e['suite']=='LIBERO50' and e['seed']==seed
 assert sha(e['bundle'])==e['bundle_sha256']=='370dd4645bb319f0f1739fa13b5a84fa82d86c7c0e0078be0da28d8caad718ae'
 for p,h in e['resource_hashes'].items():assert sha(p)==h;files[p]=h
 e['source']=str(Q/'libero50'/('seed%d'%seed)/'source')
 e['shared_pretrain_workspace']=str(Q/'libero50/shared_pretraining')
 # New own outputs only; existing assets are never relocated.
 e['output']=('/root/mlr_ifa/experiments/' if seed==41 else '/data/mlr_ifa/experiments/')+('RA_Coreset_LIBERO50_seed%d_20261021'%seed)
 assert not Path(e['output']).exists() and not Path(e['shared_pretrain_output']).exists()
 entrypath=Q/('LIBERO50_seed%d.json'%seed);wr(entrypath,e);files[str(entrypath)]=sha(entrypath)
 launcher=Q/('LIBERO50_seed%d.sh'%seed)
 launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /data/mlr_ifa/activate.sh\nexec '+PY+' '+str(OLD/'worker_entrypoint_libero50.py')+' '+str(entrypath)+'\n')
 files[str(launcher)]=sha(launcher);files[e['bundle']]=e['bundle_sha256']
 entries.append(dict(key='RA_IFA_LIBERO50_%d'%seed,method='RA_IFA',seed=seed,output=e['output'],storage_mount='/' if seed==41 else '/data',minimum_free_bytes=4500000000,argv=['bash',str(launcher)]))
for n in ('worker_entrypoint_libero50.py','worker_entrypoint.py','queue_controller.py','queue_verifier.py'):
 files[str(OLD/n)]=sha(OLD/n)
files['/data/mlr_ifa/activate.sh']=sha('/data/mlr_ifa/activate.sh')
wr(Q/'QUEUE.json',entries);wr(Q/'SOURCE_COPY_PROOF.json',proof)
for n in ('QUEUE.json','queue_controller.py','seed_supervisor.py','PROTOCOL.md','IMPLEMENTATION_NOTES.md'):
 files[str(Q/n)]=sha(Q/n)
wr(Q/'ORCHESTRATION_CONTRACT.json',dict(version='caet_v2_object_extension_libero50_v1',files=files,created=time.time(),cancelled_excluded=['COVR','A2'],old_controller_restarted=False))
with (Q/'controller.log').open('x') as log:
 p=subprocess.Popen([PY,'-u',str(Q/'queue_controller.py')],cwd=Q,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(status='DISPATCHED_NOT_YET_GPU_ACCEPTED',controller_pid=p.pid,unix=time.time(),queue=entries,orchestration_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),source_copy_exact=True,disk={x:shutil.disk_usage(x).free for x in ('/','/data')})
wr(Q/'DISPATCH_RECEIPT.json',receipt);print(json.dumps(receipt))'''

def main():
 for seed in (41,43):
  g=R/('gate1_seed%d'%seed)
  assert json.loads((g/'LOCAL_AUDIT.json').read_text())['status']=='GATE1_PASS'
  assert json.loads((g/'EVIDENCE.json').read_text())['protocol_sha256']==hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 stamp=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));stamp.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  s=c.open_sftp();s.mkdir(Q)
  uploads={n:(R/n).read_bytes() for n in ('queue_controller.py','seed_supervisor.py','PROTOCOL.md','IMPLEMENTATION_NOTES.md')}
  for seed in (41,43):
   folder='object_seed%d'%seed;s.mkdir(Q+'/'+folder)
   uploads[folder+'/GATE1_EVIDENCE.json']=(R/('gate1_seed%d'%seed)/'EVIDENCE.json').read_bytes()
   uploads[folder+'/GATE1_LOCAL_AUDIT.json']=(R/('gate1_seed%d'%seed)/'LOCAL_AUDIT.json').read_bytes()
  for n,b in uploads.items():
   with s.open(Q+'/'+n,'wb') as f:f.write(b)
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180)
  raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (stamp/'DEPLOY_ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stdout=raw.decode(),stderr=err.decode(),uploads={n:hashlib.sha256(b).hexdigest() for n,b in uploads.items()}),indent=2)+'\n',encoding='utf8')
  assert rc==0,err.decode()
  (stamp/'DISPATCH_RECEIPT.json').write_bytes(raw)
  for n in ('QUEUE.json','ORCHESTRATION_CONTRACT.json','SOURCE_COPY_PROOF.json'):
   with s.open(Q+'/'+n,'rb') as f:(stamp/n).write_bytes(f.read())
  print(raw.decode());print(stamp)
 finally:pw=None;c.close()
if __name__=='__main__':main()
