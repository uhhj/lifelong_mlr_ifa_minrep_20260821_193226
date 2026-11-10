"""Read-only local audit of both stripped-S variants; no training or rollout."""
from pathlib import Path
import ast,json,hashlib,datetime
R=Path(__file__).resolve().parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  a=R/'cases'/(suite+'_ALIGN_'+str(seed));m=R/'cases'/(suite+'_MLR_'+str(seed))
  ai=J(a/'INPUT.json');mi=J(m/'INPUT.json');assert ai['variant']=='ALIGN' and mi['variant']=='MLR'
  assert {k:v for k,v in ai.items() if k not in ['case','variant']}=={k:v for k,v in mi.items() if k not in ['case','variant']}
  for name in ['case_runtime.py','train.py','admission.py','updates.py','evaluate.py','SOURCE_HASHES.json','resolved_eval.json']:
   assert H(a/name)==H(m/name),(suite,seed,name)
  ac=J(a/'resolved_random.json');mc=J(m/'resolved_random.json')
  for name in ['experiment_dir','buffer_dir','run_root']:ac.pop(name);mc.pop(name)
  assert ac==mc
  for t in range(6,10):
   assert H(a/'manifests'/('task%d.json'%t))==H(m/'manifests'/('task%d.json'%t))
   assert H(a/('ifa_task%d.json'%t))==H(m/('ifa_task%d.json'%t))
  assert ai['production_budget']==(105500 if suite=='OBJECT' else 76100)
  for d in [a,m]:
   for p in d.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,8))
  checks.append(dict(suite=suite,seed=seed,initialization_inputs_identical=True,random_population_order_identical=True,ifa_pairs_identical=True,optimizer_scheduler_config_identical=True,full_s_evaluator_byte_identical_between_variants=True,only_objective_switch='INPUT.variant: ALIGN masked next_action vs MLR unmasked original last action'))
for f in ['updates.py','train.py','admission.py']:
 t=(R/f).read_text();tree=ast.parse(t)
 calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
 assert not calls.intersection({'probe','choose','collect','retain','retention','configure_effect','directional','tensors','component_terms'}),(f,calls)
out=dict(status='PASS_STATIC_PAIRED_PROTOCOL_NOT_GPU_ACCEPTANCE',observed=datetime.datetime.now().astimezone().isoformat(),pairs=checks,runs=12,epochs=4800,production=1089600,admission_optimizer=240,formal=2400,training_collection=0,removed=['RA coreset','E execution supervision','R retention','q support calibration','C/Q candidate gating','online training collection'],source='Actual accepted complete-S initialization/build/update backbone/evaluator; input sampling restored to original Random')
(R/'PREPARATION_VALIDATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='pairs'}))
