"""Explicit user cancellation: stop this queue tree, preserve all scientific assets."""
import datetime
import getpass
import hashlib
import json
import shlex
from pathlib import Path
import paramiko

R = Path(__file__).resolve().parent
SCRIPT = r'''
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021')
D=Q/'libero50_taskmap01'
O=Path('/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021_taskmap01')
def proc(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,ppid=int(s[1]),state=s[0],start=s[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
 except (OSError,ProcessLookupError):return None
def scan():return {int(p.name):x for p in Path('/proc').glob('[0-9]*') if (x:=proc(int(p.name)))}
def same(x):
 y=proc(x['pid']);return y is not None and y['start']==x['start'] and y['state']!='Z'
def sig(x,n):
 if same(x):
  try:os.kill(x['pid'],n)
  except ProcessLookupError:pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
pid=json.loads((D/'controller.pid.json').read_text())['pid'];root=proc(pid)
assert root and str(D/'libero50_taskmap01_controller.py') in root['command'], 'Controller identity changed; inspect before cancellation'
marker=D/'PRETRAINING_CANCELLED_BY_USER.json'
assert not marker.exists(), 'Cancellation already recorded; verify read-only instead'
before=dict(unix=time.time(),controller=root,epochs=json.loads((O/'pretraining/epochs.json').read_text()) if (O/'pretraining/epochs.json').exists() else [],gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
record=dict(status='USER_CANCEL_REQUESTED',user_instruction='放弃预训练',shared_pretraining='CANCELLED_BY_USER',downstream='NOT_STARTED_DEPENDENCY_CANCELLED_NO_AUTOSTART',scientific_completion=False,old_assets_preserved=True,before=before)
marker.write_text(json.dumps(record,indent=2)+'\n')
# Freeze scheduler first, then its entire known subtree to prevent spawning during traversal.
sig(root,signal.SIGSTOP);targets={pid:root}
for _ in range(6):
 table=scan();changed=False
 for n,x in table.items():
  if x['ppid'] in targets and n not in targets:
   targets[n]=x;sig(x,signal.SIGSTOP);changed=True
 if not changed:break
 time.sleep(.1)
assert os.getpid() not in targets
record['targets']=list(targets.values())
marker.write_text(json.dumps(record,indent=2)+'\n')
# Queue terminates before workers are released, so no following entry can launch.
sig(root,signal.SIGTERM);sig(root,signal.SIGCONT)
for n,x in targets.items():
 if n!=pid:sig(x,signal.SIGTERM);sig(x,signal.SIGCONT)
deadline=time.monotonic()+20
while time.monotonic()<deadline and any(same(x) for x in targets.values()):time.sleep(.5)
forced=[x for x in targets.values() if same(x)]
for x in forced:sig(x,signal.SIGKILL)
time.sleep(1)
remaining=[x for x in targets.values() if same(x)]
foreign=[]
for x in scan().values():
 if x['pid']==os.getpid() or 'python -c' in x['command']:continue
 if str(D) in x['command'] or 'l50_exp.' in x['command']:foreign.append(x)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
gpu_pids={int(l.split(',')[0]) for l in gpu.splitlines() if l.strip()}
assert not remaining and not foreign and not (gpu_pids & set(targets)), (remaining,foreign,gpu)
hashes={}
contract=json.loads((O/'CONTRACT.json').read_text())
for p,h in contract['files'].items():hashes[p]=sha(p)==h
for n,h in contract['source_hashes'].items():hashes['source/'+n]=sha(D/'shared_pretraining/source'/n)==h
for p,h in json.loads((D/'CONTRACT.json').read_text())['files'].items():hashes['dispatch/'+p]=sha(p)==h
assert all(hashes.values())
exits={}
for base in (D,O,D/'shared_pretraining/admission'):
 for p in base.glob('*.exit'):exits[str(p)]=p.read_text()
record.update(status='CANCELLED_BY_USER_VERIFIED_STOPPED',verified_unix=time.time(),forced_pids=[x['pid'] for x in forced],remaining=remaining,matching_processes=foreign,gpu_after=gpu,hashes=hashes,exits_preserved=exits,completed_epochs=len(before['epochs']),within_epoch_updates='unknown; partial epoch not accepted',downstream_outputs={str(Path(e['output'])):Path(e['output']).exists() for e in json.loads((D/'QUEUE.json').read_text())})
marker.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''

dest=R/('pretraining_cancelled_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ')
try:
    c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
    i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180)
    i.channel.shutdown_write();raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
    (dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8')
    assert rc==0,err.decode()
    d=json.loads(raw);(dest/'STOP_EVIDENCE.json').write_bytes(raw)
    receipt=dict(status='LOCAL_STOP_AUDIT_PASS',sha256=hashlib.sha256(raw).hexdigest(),hashes_verified=len(d['hashes']),no_target_processes=not d['remaining'] and not d['matching_processes'],no_downstream_outputs=not any(d['downstream_outputs'].values()))
    assert receipt['no_target_processes'] and receipt['no_downstream_outputs'] and all(d['hashes'].values())
    (dest/'LOCAL_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(dest);print(json.dumps(receipt))
finally:pw=None;c.close()
