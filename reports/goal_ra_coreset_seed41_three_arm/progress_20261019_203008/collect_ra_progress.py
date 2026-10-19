"""Read-only RA contract/metadata capture; never loads a checkpoint or starts an environment."""
from pathlib import Path
import json,hashlib,getpass,paramiko,sys
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
code=r'''
from pathlib import Path
import json,time,hashlib
from coreset_exp.common import O,S,verify,sha
start=time.time();verify();files={};manifest={};logs={}
names=['PARENT_BINDING.json','CONTRACT.json','MODAL_CONTRACT.json','FEATURES.json','RUN_STATUS.json','GPU_PREFLIGHT.json','OFFLINE_ACCEPTANCE.json','TASK6_TENSOR_PARITY.json','commands.jsonl','timings.jsonl','pipeline.started','pipeline.exit','supervisor.exit','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','ra_coreset/TRAINING_COMPLETE.json']
names += [str(p.relative_to(O)) for pattern in ['ra_coreset/storage/*.json','offline_selection/*.json','ra_coreset/evaluation/*.json','ra_coreset/evaluation/*.jsonl'] for p in O.glob(pattern)]
for n in names:
 p=O/n
 if not p.is_file():continue
 raw=p.read_bytes()
 if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
 assert len(raw)<8*1024**2
 files[n]=raw.decode();manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for n in ['software.log','software.exit','prepare.exit','offline.exit','freeze.exit','preflight.exit']:
 p=S.parent/n
 if p.exists():
  raw=p.read_bytes();files['admission/'+n]=raw.decode();manifest['admission/'+n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for n in ['train.log','evaluate.log']:
 p=O/n
 if p.exists():logs[n]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=p.read_text(errors='replace').splitlines()[-5:])
print(json.dumps(dict(status='PASS_RA_FROZEN_INPUTS',time=time.time(),seconds=time.time()-start,files=files,manifest=manifest,logs=logs,new_training_or_environment_calls=0)))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 cmd='source '+Q+"/ra/env.sh && /data/mlr_ifa/env/bin/python - <<'PY'\n"+code+'\nPY'
 _,a,b=c.exec_command(cmd,timeout=180);raw=a.read();err=b.read().decode();rc=a.channel.recv_exit_status()
 (D/'RA_READONLY_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 v=json.loads(raw);files=v.pop('files')
 for n,txt in files.items():
  p=D/'ra_evidence'/n;p.parent.mkdir(parents=True,exist_ok=True);raw=txt.encode();assert hashlib.sha256(raw).hexdigest()==v['manifest'][n]['sha256'];p.write_bytes(raw)
 (D/'RA_FROZEN_INPUT_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(dict(status=v['status'],seconds=v['seconds'],files=len(files),logs=v['logs']),indent=2))
finally:c.close()
