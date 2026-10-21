"""Download only an explicit lightweight allowlist after process acceptance."""
import datetime,getpass,json,hashlib,shlex,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
from pathlib import Path
import json,hashlib,os,time
W=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021/object_seed@SEED@')
O=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed@SEED@_20261021')
assert (W/'supervisor.exit').read_text().strip()=='0'
assert (O/'pipeline.exit').read_text().strip()=='0'
assert json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
residual=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if (p/'cwd').resolve() in (W,W/'source') and any(k in cmd for k in ('caet_v2_exp','coreset_exp.evaluate','recovery_environment01.py')):residual.append(int(p.name))
 except OSError:pass
assert not residual,residual
entries={}
def add(p,name):
 assert p.is_file() and p.suffix in ('.json','.jsonl','.exit')
 raw=p.read_bytes();assert len(raw)<8*1024**2
 entries[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for p in O.glob('*.json'):add(p,'current/'+p.name)
for p in O.glob('*.exit'):add(p,'current/'+p.name)
for pattern in ('ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','ra_coreset/TRAINING_COMPLETE.json','ra_coreset/ifa_task*.json','ra_coreset/storage/*.json','ra_coreset/auxiliary/*.json','ra_coreset/constraints/*.json','ra_coreset/functional_references/*.json','ra_coreset/evaluation/*.json','ra_coreset/evaluation/*.jsonl','effects/current/*/AUDIT.json','effects/current/*/SUPPORT.json','effects/models/*/ACCEPTANCE.json','effects/models/*/epochs.json','preflight/optimizer_calls.json'):
 for p in O.glob(pattern):add(p,'current/'+str(p.relative_to(O)))
for label,root in [('baseline',Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed@SEED@_20261019')),('v1',Path('/data/mlr_ifa/experiments/CAET_OBJECT_seed@SEED@_full_20261021'))]:
 for pattern in ('ra_coreset/evaluation/trials*.jsonl','ra_coreset/storage/*.json','ra_coreset/ifa_task*.json','ra_coreset/INITIAL_STATE.json'):
  for p in root.glob(pattern):add(p,label+'/'+str(p.relative_to(root)))
for n in ('supervisor.exit','ENTRY.json'):add(W/n,'launcher/'+n)
print(json.dumps(dict(status='LIGHT_MANIFEST_READY',observed_unix=time.time(),residual_workers=residual,entries=entries)))
'''
seed=int(sys.argv[1]);assert seed in (41,43)
SCRIPT=SCRIPT.replace('@SEED@',str(seed))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();assert rc==0,err.decode()
 manifest=json.loads(raw);dest=R/('final_seed%d_'%seed+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
 (dest/'PACKAGE_MANIFEST.json').write_bytes(raw)
 if '--manifest-only' in sys.argv:
  print(json.dumps(dict(directory=str(dest),status='MANIFEST_ONLY_NOT_DOWNLOADED',files=len(manifest['entries']))))
  sys.exit(0)
 s=c.open_sftp()
 for name,item in manifest['entries'].items():
  p=(dest/name).resolve();assert dest.resolve() in p.parents
  p.parent.mkdir(parents=True,exist_ok=True)
  with s.open(item['path'],'rb') as f:data=f.read()
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];p.write_bytes(data)
 print(json.dumps(dict(directory=str(dest),files=len(manifest['entries']),bytes=sum(v['bytes'] for v in manifest['entries'].values()))))
finally:pw=None;c.close()
