"""One new, isolated S-derived MLR+IFA+ALIGN experiment; local text only."""
from pathlib import Path
import json,hashlib,shutil,ast
R=Path(__file__).resolve().parent;B=R.parent/'20261109_mlr_ifa_align_baseline'
NAME='S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1';Q='/data/mlr_ifa/experiment_workspaces/'+NAME;P='/data/mlr_ifa/experiments/'+NAME
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def txt(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x,encoding='utf-8',newline='\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=read(B/'INPUT_SPEC.json');e=read(R/'preflight_readonly_20261110_201843/EVIDENCE.json');s=read(R/'capacity_readonly_20261110_202315/EVIDENCE.json')
external={};bindings={};configdiff={}
expanded=[(suite+'_'+variant+'_'+str(seed),suite+'_'+str(seed),variant) for suite in ['OBJECT','GOAL'] for seed in [41,42,43] for variant in ['ALIGN','MLR']]
for case,source_case,variant in expanded:
 meta=spec[source_case];suite,seed=source_case.split('_');seed=int(seed);dest=R/'cases'/case
 assert not (dest/'INITIAL_A.json').exists(),'Do not overwrite a real run'
 old=read(B/'cases'/source_case/'INPUT.json');parent=s['cases'][source_case]
 inp={k:old[k] for k in ['case','seed','suite','parent','labels','initial_state','baseline_gpu']}
 inp.update(case=case,variant=variant,source_case=source_case,align_parent=parent['align_parent'],align_source=parent['source_root'],production_budget=105500 if suite=='OBJECT' else 76100,admission_optimizer_calls=20,training_environment_calls=0)
 full=R.parent/('20261028_support_gate_full_s_object' if suite=='OBJECT' else '20261029_support_gate_full_s_goal')
 finals=sorted(full.glob('final_readonly_*/light/seed%d/STAGE_S_6.json'%seed));assert finals
 stage=read(finals[-1]);inp.update(full_s_task6_path=stage['model'],full_s_task6_sha256=stage['sha256'])
 put(dest/'INPUT.json',inp);put(dest/'SOURCE_HASHES.json',parent['source_hashes']);external.update(parent['source_hashes'])
 external[parent['align_parent']+'/CONTRACT.json']=parent['source_contract_sha256'];external[stage['model']]=stage['sha256']
 # Old private orchestration files are not dependencies of the new run.
 previous=read(B/'cases'/source_case/'CONTRACT.json')
 for p,h in previous['files'].items():
  if 'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1' not in p and 'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2' not in p:external[p]=h
 pops={};valid={}
 for t in range(6,10):
  m=read(B/'cases'/source_case/'manifests'/('task%d.json'%t));pops[t]=len(m['rows']);valid[t]=sum(bool(x['bc_valid']) for x in m['rows']);put(dest/'manifests'/('task%d.json'%t),m)
  oldinfo=e['cases'][source_case]['ifa'][str(t)];put(dest/('ifa_task%d.json'%t),oldinfo['data']);external[oldinfo['path']]=oldinfo['sha256']
  for row in m['rows']:external[inp['parent']+'/'+row['relative_path']]=row['sha256']
 cfg=read(B/'cases'/source_case/'resolved_random.json');scfg=parent['ra_config']
 # All actual model/optimizer/data interpretation settings remain full-S settings.
 for key in ['seed','pretrain','benchmark_name','data','policy','train','eval','lifelong','gamma','shape_meta']:
  assert cfg[key]==scfg[key],(case,key)
 cfg.update(feature_dir=inp['parent']+'/features',buffer_dir=P+'/'+case+'/A/logical_buffer_empty',experiment_dir=P+'/'+case+'/A/models',run_root=P+'/'+case+'/A')
 put(dest/'resolved_random.json',cfg);put(dest/'resolved_eval.json',parent['eval_config'])
 txt(dest/'layout.py',"from pathlib import Path\nARMS='A'\nSUITE=%r\nSEED=%r\nNAME=%r\nWORKSPACE=Path(%r)\nDATA_PRIVATE=Path(%r)\nOUTPUT={'A':DATA_PRIVATE/%r/'A'}\nPOP=%r\nVALID=%r\ndef model_path(arm,stage):return OUTPUT[arm]/('stage%%d.pth'%%stage)\n"%(suite,seed,NAME,Q,P,case,pops,valid))
 txt(dest/'case_runtime.py',(R/'case_runtime_template.py').read_text(encoding='utf-8'))
 ev=(full/('seed%d'%seed)/'evaluate.py').read_text(encoding='utf-8')
 assert "from runtime import *" in ev and "load('S',6" in ev and "js(O/'resolved_eval.json')" in ev
 ev=ev.replace('from runtime import *','from case_runtime import *').replace("load('S',6","load('A',6").replace("js(O/'resolved_eval.json')","js(R/'resolved_eval.json')")
 txt(dest/'evaluate.py',ev)
 for n in ['train.py','admission.py','updates.py']:txt(dest/n,(R/n).read_text(encoding='utf-8'))
 bindings[case]=dict(variant=variant,source_case=source_case,full_s_source=str(full/('seed%d'%seed)),evaluate_parent_sha256=digest(full/('seed%d'%seed)/'evaluate.py'),evaluate_changes=['runtime import renamed','arm S renamed A','own frozen eval config path'],random_manifest_sources={str(t):digest(B/'cases'/source_case/'manifests'/('task%d.json'%t)) for t in range(6,10)},random_ifa_sources={str(t):e['cases'][source_case]['ifa'][str(t)]['sha256'] for t in range(6,10)},production=inp['production_budget'],population=pops,valid=valid,full_s_task6=stage)
put(R/'EXTERNAL_FILES.json',external);put(R/'PARENT_BINDINGS.json',bindings)
for n in ['BASELINE_BINDING.json','BASELINE_OUTCOMES.json']:
 if (B/n).exists():shutil.copyfile(B/n,R/n)
for p in (R/'cases').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
put(R/'PREPARATION_STATUS.json',dict(status='PREPARED_NOT_FROZEN_NOT_DEPLOYED',cases=list(bindings),objective='MLR: original_BC+0.1*IFA; ALIGN: aligned_BC+0.1*IFA',epochs=4800,production=1089600,admission=240,formal=2400,training_collection=0,external_files=len(external)))
print(json.dumps(dict(status='PREPARED_NOT_DEPLOYED',cases=list(bindings),external_files=len(external))))
