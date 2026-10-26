"""Suite/seed-only adaptation from accepted S GOAL42; does not dispatch."""
from pathlib import Path
import ast,json,hashlib,difflib,sys
R=Path(__file__).resolve().parent;P=R.parent/'20261026_ccbp_v4_s_goal42_43';T=P/'seed42'
NAME='CCBP_V4_S_OBJECT41_42_43_20261026_v1';Q='/data/mlr_ifa/experiment_workspaces/'+NAME
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
js=lambda p:json.loads(p.read_text(encoding='utf8'))
pre=js(Path(sys.argv[1])/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY'
assert js(P/'RUN_STATUS.json')['status']=='COMPLETE_ACCEPTED_FINITE_EXTENSION'
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
review={};bundle={}
names=['layout.py','runtime.py','legacy_losses.py','losses.py','memory.py','updates.py','initialization.py','train.py','collect.py','evaluate.py','synthetic_checks.py']
for seed in (41,42,43):
 parent=pre['parents'][str(seed)];mapping={}
 for n in names:
  original=(T/n).read_text(encoding='utf8');s=original
  if n=='layout.py':
   s=s.replace('SEED=42','SEED='+str(seed)).replace('CCBP_V4_S_GOAL42_43_20261026_v1',NAME).replace("'/data':4*1024**3","'/data':6*1024**3")
   for key,field in [('POP','population'),('VALID','valid'),('OLD','old_valid')]:
    old=next(x for x in s.splitlines() if x.startswith(key+'='));s=s.replace(old,key+'='+repr({int(k):v[field] for k,v in parent['stages'].items()}))
  if n=='runtime.py':
   s=s.replace("O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed42_20261022')",'O=Path('+repr(parent['output'])+')').replace("S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_42/source')",'S=Path('+repr(parent['source'])+')').replace("CORESET_SEED='42'","CORESET_SEED="+repr(str(seed)))
  if n=='train.py':s=s.replace('a.step==76100','a.step==105500')
  if n in ['losses.py','legacy_losses.py','memory.py','updates.py','initialization.py','collect.py','evaluate.py','synthetic_checks.py']:assert s==original
  ast.parse(s,feature_version=(3,8));mapping[n]=s
  review[f'seed{seed}/{n}']=dict(parent_sha256=sha(T/n),new_text_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if s!=original:put(R/'review'/f'seed{seed}_{n}.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='accepted_GOAL42/'+n,tofile=f'OBJECT{seed}/'+n)))
 mapping['INITIAL_SPLIT.json']=json.dumps(parent['initial_split'],indent=2)+'\n';mapping['PARENT_BINDING.json']=json.dumps(parent,indent=2)+'\n';mapping['PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
 for n,s in mapping.items():put(R/f'seed{seed}'/n,s);bundle[f'seed{seed}/{n}']=s
 f=(P/'final_seed42_readonly.py').read_text(encoding='utf8');start=f.index('Q=Path(');end=f.index('\nsys.path.insert',start)
 f=f[:start]+f"Q=Path('{Q}/seed{seed}');O=Path('{parent['output']}');S=Path('{parent['source']}')"+f[end:];f=f.replace('==76100','==105500').replace("fp.name!='CONTRACT.json'","fp.name not in ['CONTRACT.json','PARENT_BINDING.json']")
 ast.parse(f,feature_version=(3,8));put(R/f'final_seed{seed}_readonly.py',f)
for n in ['controller.py','PROTOCOL.md']:bundle[n]=(R/n).read_text(encoding='utf8')
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(Path(sys.argv[1])/'EVIDENCE.json'),seed41_contract=pre['seed41_contract'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
put(R/'DEPLOY_TEXT_BUNDLE.json',json.dumps(bundle,ensure_ascii=True,indent=2)+'\n')
put(R/'UPLOAD_REVIEW.json',json.dumps(dict(status='REVIEWED_TEXT_ONLY',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},parent_implementation=review,excludes='credentials/.session_control/private payloads/archives'),indent=2)+'\n')
print('Built finite OBJECT S three-seed sources; no dispatch.')
