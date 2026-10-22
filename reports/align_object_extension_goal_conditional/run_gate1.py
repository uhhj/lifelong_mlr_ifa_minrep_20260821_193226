import datetime,getpass,json,hashlib,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
source=(R/'bc_labels.py').read_text(encoding='utf8')+'\n'+(R/'gate1_readonly.py').read_text(encoding='utf8')
out=R/('gate1_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
(out/'PRECOMMIT.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2)+'\n',encoding='utf8')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command("env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/mlr_ifa/env/bin/python -c "+shlex.quote(source),timeout=900)
 raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 data=json.loads(raw);assert data['status']=='PASS' and not data['cuda_initialized']
 legacy=json.loads((R.parent/'20261022_caet_v3_tac_eftr_object41/GATE1_EVIDENCE.json').read_text(encoding='utf8'))
 for t,new in data['runs']['OBJECT_41']['tasks'].items():
  old=legacy['tasks'][t];assert new['valid_bc_labels']==old['valid_bc_labels']
  assert {x['sample']:x['next_action_sha256'] for x in new['labels_metadata']}=={x['sample']:x['next_action_sha256'] for x in old['labels_metadata']}
 summary={}
 for key,v in data['runs'].items():
  rawrun=json.dumps(v,indent=2).encode();p=out/(key+'.json');p.write_bytes(rawrun)
  audit=dict(status='PASS',evidence_sha256=hashlib.sha256(rawrun).hexdigest(),tasks={t:{k:x[k] for k in ['total_windows','valid_bc_labels','invalid']} for t,x in v['tasks'].items()},legacy_OBJECT41_BC_mapping_exact=True)
  (out/(key+'_LOCAL_AUDIT.json')).write_text(json.dumps(audit,indent=2)+'\n');summary[key]=audit
 (out/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS',runs=summary,protocol_sha256=hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest(),optimizer_calls=0,environment_steps=0),indent=2)+'\n')
 (R/'GATE1_INDEX.json').write_text(json.dumps(dict(directory=out.name,runs=summary),indent=2)+'\n')
 print(out);print(json.dumps(summary,indent=2))
finally:pw=None;c.close()
