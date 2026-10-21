"""Fixed read-only mechanism ledger audit; no model loads or environment calls."""
import datetime,getpass,hashlib,json,shlex,sys
from pathlib import Path
import paramiko
SCRIPT=r'''
import json,hashlib,time
from pathlib import Path
O=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021')
h=json.loads((O/'ra_coreset/epochs.json').read_text());out={}
for t in (8,9):
 root=O/'effects/models'/('task%d'%t);p=root/'ACCEPTANCE.json'
 if not p.exists():continue
 a=json.loads(p.read_text());support=O/'effects/current'/('task%d'%t)/'SUPPORT.json'
 item=dict(admission=a,support_sha256=hashlib.sha256(support.read_bytes()).hexdigest(),epochs=[])
 for ep in [r for r in h if r['task']==t]:
  n='task%d_epoch%02d.json'%(t,ep['epoch']);p=O/'ra_coreset/auxiliary'/n;raw=p.read_bytes();rows=json.loads(raw)
  c=O/'ra_coreset/constraints'/n
  item['epochs'].append(dict(epoch=ep['epoch'],batches=ep['batches'],steps=ep['steps'],rows=rows,sha256=hashlib.sha256(raw).hexdigest(),constraints=json.loads(c.read_text())))
 out[str(t)]=item
print(json.dumps(dict(observed_unix=time.time(),tasks=out)))
'''
p=Path(sys.argv[1]);c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
 (p/'MECHANISM_ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0
 d=json.loads(raw)
 # Split by task into lightweight outcome/gradient ledgers, no input payloads.
 receipt={}
 for t,v in d['tasks'].items():
  raw=json.dumps(v).encode();assert len(raw)<8*1024**2;n='MECHANISM_task'+t+'.json';(p/n).write_bytes(raw);receipt[n]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 (p/'MECHANISM_RECEIPT.json').write_text(json.dumps(dict(observed_unix=d['observed_unix'],files=receipt),indent=2),encoding='utf8')
 print(json.dumps(receipt))
finally:pw=None;c.close()
