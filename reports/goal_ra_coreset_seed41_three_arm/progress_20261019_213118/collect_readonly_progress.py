"""Collect immutable-in-memory metadata/outcome snapshots and verify frozen inputs."""
from pathlib import Path
import json,hashlib,getpass,paramiko,datetime,sys
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
code=r'''
from pathlib import Path
import json,time,hashlib
from coreset_exp.common import O,verify,sha
start=time.time();c=verify();files={};manifest={};cp={};logs={}
names=['commands.log','FEATURES.json','TASK6_TENSOR_PARITY.json','export.log','pipeline.started','pipeline.pid']
for arm in ['random','coreset']:
 names += [arm+'/'+n for n in ['epochs.json','INITIAL_STATE.json','TRAINING_COMPLETE.json','buffer_after_task9.json']]
 names += [str(p.relative_to(O)) for pattern in ['storage/*.json','evaluation/*.json','evaluation/*.jsonl'] for p in (O/arm).glob(pattern)]
 for p in (O/arm/'models').glob('*ep99.pth'):cp[str(p.relative_to(O))]=dict(bytes=p.stat().st_size,sha256=sha(p))
 for p in (O/arm/'recovery').glob('*.pth'):logs[str(p.relative_to(O))]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
for n in names:
 p=O/n
 if not p.is_file():continue
 raw=p.read_bytes()
 if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
 assert len(raw)<8*1024**2
 files[n]=raw.decode();manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for name in ['train_random.log','train_coreset.log','evaluate_random.log','evaluate_coreset.log']:
 p=O/name
 if p.exists():logs[name]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=p.read_text(errors='replace').splitlines()[-8:])
out=dict(status='PASS_READONLY_FROZEN_INPUTS',time=time.time(),seconds=time.time()-start,contract_sha256=sha(O/'CONTRACT.json'),source_config_input_modal_features_orchestration_verified=True,checkpoints=cp,logs=logs,files=files,manifest=manifest,new_optimizer_or_environment_calls=0)
print(json.dumps(out))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 cmd='source '+Q+"/baseline/source/coreset_exp/env.sh && /data/mlr_ifa/env/bin/python - <<'PY'\n"+code+'\nPY'
 _,a,b=c.exec_command(cmd,timeout=180);out=a.read();err=b.read().decode();rc=a.channel.recv_exit_status()
 (D/'READONLY_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 v=json.loads(out);files=v.pop('files')
 for n,txt in files.items():
  p=D/'baseline_evidence'/n;p.parent.mkdir(parents=True,exist_ok=True);raw=txt.encode();assert hashlib.sha256(raw).hexdigest()==v['manifest'][n]['sha256'];p.write_bytes(raw)
 (D/'FROZEN_INPUT_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(dict(status=v['status'],seconds=v['seconds'],files=len(files),checkpoints=len(v['checkpoints']),logs=v['logs']),indent=2))
finally:c.close()
