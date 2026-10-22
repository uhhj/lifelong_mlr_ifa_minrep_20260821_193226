"""Copy completed-epoch JSON ledgers only, no model/optimizer/environment calls."""
import datetime,getpass,json,hashlib,shlex,argparse
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,choices=[7,8,9]);args=ap.parse_args()
SCRIPT=r'''
import json,time,hashlib
from pathlib import Path
O=Path('/data/mlr_ifa/experiments/CAETv3_EFTR_OBJECT_seed41_20261022_collate01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
history=json.loads((O/'ra_coreset/epochs.json').read_text());files={};manifest={}
stage_filter=STAGE_FILTER
if stage_filter is not None:
 history=[h for h in history if h['task']==stage_filter]
 assert history and history[0]['epoch']==0
def add(p):
 raw=p.read_bytes();assert p.suffix=='.json' and len(raw)<8*1024**2
 name=str(p.relative_to(O));files[name]=raw.decode();manifest[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
stages=sorted({x['task'] for x in history if x['task']>6});binding={}
for t in stages:
 add(O/'ra_coreset/dataset_manifests'/('task%d.json'%t))
 for h in history:
  if h['task']!=t:continue
  for folder in ('auxiliary','constraints'):add(O/'ra_coreset'/folder/('task%d_epoch%02d.json'%(t,h['epoch'])))
 for j in range(6,t):
  p=O/'teachers'/('task%d.json'%j);add(p);meta=json.loads(p.read_text())
  add(O/'ra_coreset/storage'/('task%d.json'%j))
  actual=dict(teacher_sha256=sha(p.with_suffix('.pth')),checkpoint_sha256=sha(O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%j)),storage_sha256=sha(O/'ra_coreset/storage'/('task%d.json'%j)))
  assert actual['teacher_sha256']==meta['sha256'] and actual['checkpoint_sha256']==meta['checkpoint_sha256'] and actual['storage_sha256']==meta['storage_sha256']
  binding[str(j)]=actual
print(json.dumps(dict(observed_unix=time.time(),history=history,teacher_binding=binding,manifest=manifest,files=files)))
'''
SCRIPT=SCRIPT.replace('STAGE_FILTER',repr(args.task))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
 d=json.loads(raw);p=R/('eftr_mechanism_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 for name,txt in d.pop('files').items():
  target=(p/name).resolve();assert p.resolve() in target.parents
  data=txt.encode();assert hashlib.sha256(data).hexdigest()==d['manifest'][name]['sha256'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 d.update(server_writes=0,optimizer_calls=0,model_forwards=0,environment_steps=0)
 (p/'MANIFEST.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf8');print(p)
finally:pw=None;c.close()
