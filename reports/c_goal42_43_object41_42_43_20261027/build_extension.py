"""Suite/seed/layout-only adaptation of accepted C; no experiment dispatch."""
from pathlib import Path
import ast,json,hashlib,difflib,sys
R=Path(__file__).resolve().parent;P=R.parent/'20261025_ccbp_v3_full'
NAME='CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1';Q='/data/mlr_ifa/experiment_workspaces/'+NAME
CASES=['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
prepath=Path(sys.argv[1]);pre=json.loads((prepath/'EVIDENCE.json').read_text(encoding='utf8'));assert pre['status']=='PASS_PREDEPLOY_READONLY'
bundle={};review={};finals={}
for case in CASES:
 parent=pre['parents'][case];seed=parent['seed'];steps=76100 if parent['suite']=='GOAL' else 105500
 for n in ['layout.py','runtime.py','legacy_losses.py','losses.py','directional_prototype.py','updates.py','train.py','evaluate.py','synthetic_checks.py']:
  assert hashlib.sha256((P/n).read_text(encoding='utf8').encode()).hexdigest()==pre['c_implementation'][n],n
  old=(P/n).read_text(encoding='utf8');s=old
  if n=='layout.py':
   s=s.replace("ARMS='CRG'","ARMS='C'\nSEED="+str(seed)).replace('CCBP_V3_FULL_GOAL41_20261025_v1_engineering02',NAME)
   s=s.replace("OUTPUT={a:DATA_PRIVATE/a for a in ARMS}","CASE_PRIVATE=DATA_PRIVATE/"+repr(case)+"\nOUTPUT={a:CASE_PRIVATE/a for a in ARMS}")
   s=s.replace("int(2.50*1024**3)",'6*1024**3').replace("DATA_PRIVATE/'stage_models'","CASE_PRIVATE/'stage_models'").replace("DATA_PRIVATE/'teachers'","CASE_PRIVATE/'teachers'")
   for key,field in [('POP','population'),('VALID','valid'),('OLD','old_valid')]:
    line=next(x for x in s.splitlines() if x.startswith(key+'='));s=s.replace(line,key+'='+repr({int(t):x[field] for t,x in parent['stages'].items()}))
  if n=='runtime.py':
   for key,value in [('O',parent['output']),('S',parent['source'])]:
    line=next(x for x in s.splitlines() if x.startswith(key+'=Path('));s=s.replace(line,key+'=Path('+repr(value)+')')
   s=s.replace("CORESET_SEED='41'","CORESET_SEED="+repr(str(seed)))
  if n=='train.py':s=s.replace('a.step==76100','a.step=='+str(steps))
  if n=='evaluate.py':
   s=s.replace('Exactly 600 paired episodes','Exactly 200 paired episodes').replace('control_seed(41)','control_seed(SEED)').replace('41*1000','SEED*1000').replace('41+group','SEED+group').replace('budget=600','budget=200').replace('len(records)==600','len(records)==200').replace('episodes=600,vectors=60,cells=30','episodes=200,vectors=20,cells=10')
  if n in ['legacy_losses.py','losses.py','directional_prototype.py','updates.py','synthetic_checks.py']:assert old==s
  ast.parse(s,feature_version=(3,8));put(case+'/'+n,s);bundle[case+'/'+n]=s
  review[case+'/'+n]=dict(parent_local_bytes_sha256=sha(P/n),parent_remote_lf_sha256=pre['c_implementation'][n],sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==old)
  if s!=old:put('review/'+case+'_'+n+'.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='accepted_C/'+n,tofile=case+'/'+n)))
 for n,obj in [('PARENT_BINDING.json',parent),('INITIAL_SPLIT.json',parent['initial_split'])]:
  s=json.dumps(obj,indent=2)+'\n';put(case+'/'+n,s);bundle[case+'/'+n]=s
 bundle[case+'/PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
 # Adapt only external audit constants and remove old multi-arm predecessor check.
 f=(P/'final_readonly.py').read_text(encoding='utf8');a=f.index('Q=Path(');b=f.index('\nsys.path.insert',a)
 f=f[:a]+f"Q=Path('{Q}/{case}');O=Path('{parent['output']}');S=Path('{parent['source']}')"+f[b:]
 f=f.replace("(Q/'controller.exit')","(Q.parent/'controller.exit')").replace("str(Q) not in cmd","str(Q.parent) not in cmd")
 f=f.replace("parent_acceptance=js(Q/'PARENT_LOCAL_ACCEPTANCE.json')\nassert parent_acceptance['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'","assert (Q/'train_C.exit').read_text()=='0' and (Q/'evaluate.exit').read_text()=='0'")
 f=f.replace('==76100','=='+str(steps)).replace('len(records)==600','len(records)==200').replace('len(seen)==600','len(seen)==200').replace('episodes=600','episodes=200').replace('20261024+41000','20261024+SEED*1000')
 f=f.replace("fp.name!='CONTRACT.json'","fp.name not in ['CONTRACT.json','PARENT_BINDING.json']")
 # Independently verify exact sampler and full final-state integrity.
 f=f.replace("for stage in range(6,10):\n  cp=model_path", "for i,h in enumerate(hist):\n  stage=6+i//100;order=torch.randperm(POP[stage],generator=torch.Generator().manual_seed(i%100)).numpy().astype(np.int64)\n  assert hashlib.sha256(order.tobytes()).hexdigest()==h['sampler_sha256']\n for stage in range(6,10):\n  cp=model_path")
 f=f.replace("strict[arm]['recovery']=", "assert rec['state']['optimizer']['state'] and rec['state']['optimizer']['param_groups'] and rec['state']['scheduler']\n assert rec['manifest_sha256']==sha(O/'ra_coreset/dataset_manifests/task9.json')\n norm_modes=lambda names:{'' if n=='module' else n[7:] if n.startswith('module.') else n for n in names}\n assert set(dict(p.named_modules()))<=norm_modes(rec['state']['modes'])\n def finite(x):\n  if torch.is_tensor(x):assert torch.isfinite(x).all()\n  elif isinstance(x,dict):\n   for v in x.values():finite(v)\n  elif isinstance(x,(tuple,list)):\n   for v in x:finite(v)\n finite(rec['state']['optimizer']);finite(rec['state']['grads'])\n strict[arm]['recovery']=")
 ast.parse(f,feature_version=(3,8));put('final_'+case+'_readonly.py',f);finals[case]=f
for n in ['controller.py','PROTOCOL.md']:bundle[n]=(R/n).read_text(encoding='utf8')
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],c_contract=pre['c_contract'],seed41_contract=pre['seed41_contract'],preflight_sha256=sha(prepath/'EVIDENCE.json'),capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
put('DEPLOY_TEXT_BUNDLE.json',json.dumps(bundle,ensure_ascii=True,indent=2)+'\n')
put('UPLOAD_REVIEW.json',json.dumps(dict(status='REVIEWED_TEXT_ONLY',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review),indent=2)+'\n')
template=(R/'final_template.py').read_text(encoding='utf8');put('final_readonly.py','SCRIPTS='+ascii(finals)+'\n'+template)
print('Built C-only five-run extension. No dispatch; final wrappers still pending actual acceptance.')
