"""One engineering continuation, checked stopped, no scientific epoch repeated."""
import os,json,hashlib,subprocess,time,fcntl
from pathlib import Path
import torch
OLD=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1');Q=Path(str(OLD)+'_engineering02')
P=Path('/root/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1/B');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not Q.exists() and (OLD/'controller.exit').read_text()=='1'
old_contract=sha(OLD/'CONTRACT.json');assert old_contract=='c5aaab9b51f4b8f2f4377c6b721785cf4b7d618ef7bb2f5e90d93dedbe7ad053'
c=json.loads((OLD/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items());assert all(sha(Path(c['source_root'])/n)==h for n,h in c['parent_source'].items())
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 assert str(OLD) not in cmd
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
rec=torch.load(P/'recovery.pth',map_location='cpu');assert rec['stage']==6 and rec['next_epoch']==100 and rec['steps']==10000 and rec['contract']==old_contract
cp=Path('/root/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1/stage_models/B/stage6.pth');model=torch.load(cp,map_location='cpu')['state_dict']
assert len(model)==100 and all(torch.equal(model[k],v) for k,v in rec['state']['model'].items())
assert set(rec['state'])=={'model','optimizer','scheduler','grads','modes','requires','rng','step'} and set(rec['state']['rng'])=={'python','numpy','cpu','cuda'}
assert rec['history']==json.loads((OLD/'EPOCHS_B.json').read_text()) and len(rec['history'])==100
assert json.loads((OLD/'TASK6_PARITY_B.json').read_text())['equal'] and not list(OLD.glob('TEACHER_*.json'))
assert not Path('/data/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1/teachers').exists()
lock=open('/data/mlr_ifa/CCBP_V2_FULL_GOAL41_20261024.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,s in BUNDLE.items():
 assert '/' not in n
 if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir();files=dict(c['files'])
for n,s in BUNDLE.items():(Q/n).write_text(s);files[str(Q/n)]=sha(Q/n)
inherit=['INITIAL_B.json','ADMISSION_B_6.json','EPOCHS_B.json','STAGE_B_6.json','TASK6_PARITY_B.json']
for n in inherit:(Q/n).write_bytes((OLD/n).read_bytes());files[str(OLD/n)]=sha(OLD/n)
binding=dict(status='CPU_PREFLIGHT_PASS_PENDING_OWN_GPU_RESTORE',previous_root=str(OLD),previous_contract=old_contract,recovery_path=str(P/'recovery.pth'),recovery_sha256=sha(P/'recovery.pth'),recovery_bytes=(P/'recovery.pth').stat().st_size,completed_stage6_cp=str(cp),completed_stage6_sha256=sha(cp),completed_epochs=100,steps=10000,inherit={n:sha(OLD/n) for n in inherit},failure_sha256=sha(OLD/'FAILURE.json'),source_and_input_unchanged=True,raw_copy_created=False,reason_no_duplicate_payload='Original recovery remains intact until exact restored continuation commits next full epoch; model-only Stage6 checkpoint retained permanently.',fields=sorted(rec['state']),cpu_model_tensor_exact=True)
(Q/'REPAIR_CONTINUATION.json').write_text(json.dumps(binding,indent=2));files[str(Q/'REPAIR_CONTINUATION.json')]=sha(Q/'REPAIR_CONTINUATION.json');files[str(OLD/'CONTRACT.json')]=old_contract;files[str(OLD/'FAILURE.json')]=sha(OLD/'FAILURE.json');files[str(cp)]=sha(cp)
(Q/'CONTRACT.json').write_text(json.dumps(dict(c,files=files,created=time.time(),engineering_repair='Device-comparison normalization and exact Task6 continuation; no scientific change'),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='CONTINUATION_DISPATCHED_PENDING_GPU',root=str(Q),controller=p.pid,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),cpu_preflight=binding,observed=time.time())))
