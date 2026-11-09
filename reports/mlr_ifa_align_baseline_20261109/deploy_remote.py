"""One-shot entry: unreachable until accepted/published q1 closure is supplied."""
import json,hashlib,os,subprocess,time,ast
from pathlib import Path
NAME='MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME
P=Path('/data/mlr_ifa/experiments')/NAME
assert 'POST_Q1_ACCEPTANCE.json' in BUNDLE,'Missing independently accepted/published q1 closure; no writes'
scope={};exec(compile(BUNDLE['runtime.py'],'prepared/runtime.py','exec'),scope)
scope['previous_finished']()
a=json.loads(BUNDLE['POST_Q1_ACCEPTANCE.json'])
assert a['status']=='PASS_BOTH_Q1_SUITES_ACCEPTED_AND_PUBLISHED' and a['global_contract']==scope['OLD_HASH']
for suite in ['OBJECT','GOAL']:
 row=a['suites'][suite];assert row['audit']['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and row['audit']['formal_episodes']==600
 assert hashlib.sha256(row['audit_text'].encode()).hexdigest()==row['audit_sha256']
 assert json.loads(row['audit_text'])==row['audit']
 assert row['audit']['contract']==scope['sha'](scope['OLD'].parent/('S_Q1_'+suite+'41_42_43_20261109_rerun_v1')/'CONTRACT.json')
 assert a['published_files'][row['publication_path']]['sha256']==row['audit_sha256']
assert not Q.exists() and not P.exists(),'Unique run already exists; never redeploy'
capacity=scope['capacity']()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU must be idle after q1 acceptance'
global_contract=json.loads(BUNDLE['CONTRACT.json'])
for rel,h in global_contract['code_files'].items():assert hashlib.sha256(BUNDLE[rel].encode()).hexdigest()==h
for rel,h in global_contract['private_files'].items():assert hashlib.sha256(BUNDLE['private/'+rel].encode()).hexdigest()==h
for name,text in BUNDLE.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 if name.endswith('.py'):compile(text,name,'exec')
# Verify every old immutable input before any deployment writes.
checked={}
for case in scope['CASES']:
 c=json.loads(BUNDLE['private/'+case+'/CONTRACT.json'])
 for path,h in c['files'].items():
  if path.startswith(str(P)+'/'):continue
  if path not in checked:checked[path]=scope['sha'](path)
  assert checked[path]==h,('Parent input changed',path)
Q.mkdir();P.mkdir()
for name,text in BUNDLE.items():
 dest=P/name[len('private/'):] if name.startswith('private/') else Q/name
 dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('x') as f:f.write(text)
for case in scope['CASES']:
 d=P/case;inp=scope['js'](d/'INPUT.json')
 (d/'features').symlink_to(Path(inp['parent'])/'features',target_is_directory=True)
 (d/'random/buffer').mkdir(parents=True);(d/'random/models').mkdir();(d/'random/recovery').mkdir()
scope['verify']();scope['acceptance_gate']()
with (Q/'controller.log').open('x') as f:
 proc=subprocess.Popen([scope['PY'],'-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
scope['write'](Q/'DEPLOYMENT.json',dict(controller=proc.pid,observed=time.time(),contract_sha256=scope['sha'](Q/'CONTRACT.json'),capacity=capacity,previous_q1_publication=a['remote_commit']))
print(json.dumps(dict(status='DEPLOYED_ONCE_PENDING_ACTUAL_OWN_GPU_ADMISSION',controller=proc.pid,root=str(Q),private_root=str(P),contract_sha256=scope['sha'](Q/'CONTRACT.json'))))
