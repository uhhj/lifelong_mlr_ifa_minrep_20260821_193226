"""One-shot resource-link repair and finite continuation; preserve all original failures."""
from pathlib import Path
import sys,json,hashlib,subprocess,time,os
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v1'
sys.path.insert(0,str(Q));import runtime as rt
P=rt.P
assert not E.exists(),'Existing recovery attempt: inspect, do not repeat'
assert not rt.processes([Q,P]);rt.verify()
assert rt.sha(Q/'CONTRACT.json')=='ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f'
assert (Q/'controller.exit').read_text().strip()=='1' and rt.sha(Q/'FAILURE.json')=='b74b059b2c377de0091988bb77db57dbfa0bb710586d55ddce682cde2a024b0e'
assert rt.sha(P/'OBJECT_41/evaluate.log')=='a9df7c1cb170d335655e3e1b04faf9445090057082f252dbb14cb93f5741b112'
pre=json.loads(BUNDLE['PREFLIGHT_BINDING.json']);assets=json.loads(BUNDLE['ASSETS.json']);A=Path(assets['root'])
assert {str(p.relative_to(A)):rt.sha(p) for p in A.rglob('*') if p.is_file()}==assets['manifest']
assert not any(Path(n).suffix in ['.py','.yaml','.sh'] for n in assets['manifest'])
failed=P/'OBJECT_41/random/evaluation'
assert {str(p.relative_to(failed)):rt.sha(p) for p in failed.rglob('*') if p.is_file()}==pre['failed_evaluation_hashes']
preserved={};training_before={}
for case in rt.CASES:
 d=P/case
 target=Q/'cases'/case/'source/libero/libero/assets';assert not target.exists() and not target.is_symlink()
 if case.startswith('OBJECT_'):
  assert (d/'train.exit').read_text().strip()=='0' and len(rt.js(d/'random/epochs.json'))==400
  cp=rt.js(d/'random/TRAINING_COMPLETE.json')['checkpoints']
  assert {p.name:rt.sha(p) for p in (d/'random/models').glob('*.pth')}==cp
  for p in d.rglob('*'):
   if not p.is_file() or p.is_symlink() or 'evaluation' in p.relative_to(d).parts:continue
   if p.name!='RUN_STATUS.json' and p.suffix in ['.json','.jsonl','.exit','.log'] and not any(z in p.relative_to(d).parts for z in ['source','features','manifests']):preserved[str(p)]=rt.sha(p)
  for p in list((d/'random/models').glob('*.pth'))+[d/'random/recovery/latest.pth']:preserved[str(p)]=rt.sha(p)
  training_before[case]=dict(epochs=400,model_hashes=cp)
 else:
  assert not (d/'random/recovery/latest.pth').exists() and not (d/'train.exit').exists() and not (d/'train.log').exists()
for p in [Q/'FAILURE.json',Q/'controller.exit',Q/'EVENTS.jsonl',Q/'controller.log',P/'OBJECT_41/evaluate.log',P/'OBJECT_41/evaluate.exit']:preserved[str(p)]=rt.sha(p)
capacity=rt.capacity()
for n,t in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 if n.endswith('.py'):compile(t,n,'exec')
c=json.loads(BUNDLE['CONTRACT.json'])
assert all(hashlib.sha256(BUNDLE[n].encode()).hexdigest()==h for n,h in c['files'].items())
E.mkdir()
for n,t in BUNDLE.items():
 with (E/n).open('x') as f:f.write(t)
for case in rt.CASES:(Q/'cases'/case/'source/libero/libero/assets').symlink_to(A,target_is_directory=True)
# Preserve the exact failed attempt (including saved RNG) outside canonical evaluation output.
os.rename(failed,E/'failed_evaluation_OBJECT41')
for n,h in pre['failed_evaluation_hashes'].items():preserved[str(E/'failed_evaluation_OBJECT41'/n)]=h
rt.write(E/'PRESERVED.json',dict(files=preserved,training=training_before,original_exit=1,failed_attempt_retained=True))
rt.verify()
for case in rt.CASES:
 source=Q/'cases'/case/'source'
 actual={str(p.relative_to(source)):rt.sha(p) for p in source.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.sh'] and '__pycache__' not in p.parts}
 assert actual==rt.js(P/case/'CONTRACT.json')['source_sha256'],case
sys.path.insert(0,str(E));from resume_runtime import verify
verify()
with (E/'controller.log').open('x') as f:
 p=subprocess.Popen([rt.PY,'-B',str(E/'controller_resume.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rt.write(E/'DEPLOYMENT.json',dict(status='STARTED_PENDING_RNG_AND_ROLLOUT',controller=p.pid,contract_sha256=rt.sha(E/'CONTRACT.json'),observed=time.time(),capacity=capacity,old_controller_exit=1,training_signals=0,training_repeated=0))
print(json.dumps(dict(status='STARTED_PENDING_RNG_AND_ROLLOUT',controller=p.pid,engineering_root=str(E),contract_sha256=rt.sha(E/'CONTRACT.json'))))
