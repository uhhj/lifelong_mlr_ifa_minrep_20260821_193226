"""Prepare immutable configs/manifests from actual CPU audit; no remote writes."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
NAME='MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1'
Q='/data/mlr_ifa/experiment_workspaces/'+NAME;P='/data/mlr_ifa/experiments/'+NAME
def js(p):return json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=js(R/'capacity_readonly_20261109_164643/EVIDENCE.json');old=js(R/'preflight_readonly_20261109_163522/EVIDENCE.json');spec=js(R/'INPUT_SPEC.json')
assert e['status']=='PASS_SIX_CASE_RANDOM_ALIGN_INPUT_AUDIT'
summary={};cfgreview={}
for case,x in e['cases'].items():
 d=R/'cases'/case;dest=P+'/'+case;parents=old['cases'][case]['metadata']
 storage={t:{k:v[k] for k in ['K','before','selected']} for t,v in x['storage'].items()}
 inp=dict(case=case,seed=int(case[-2:]),suite=case.split('_')[0],parent=spec[case]['parent'],labels=x['labels'],storage=storage,initial_state=x['initial_state'],baseline_gpu=spec[case]['baseline']['historical_gpu'],all_private_outputs=P,production_budget=x['production_budget'],admission_optimizer_calls=20,training_environment_calls=0)
 put(d/'INPUT.json',inp)
 for t,m in x['manifests'].items():put(d/'manifests'/('task'+t+'.json'),m)
 put(d/'FEATURES.json',json.loads(parents['FEATURES.json']['text']))
 cfgreview[case]={}
 for arm in ['random','eval']:
  c=json.loads(parents['resolved_'+arm+'.json']['text']);original=dict(c)
  c.update(feature_dir=dest+'/features',buffer_dir=dest+'/'+arm+'/buffer',experiment_dir=dest+'/'+arm+'/models',run_root=dest+'/'+arm,contract=dest+'/CONTRACT.json')
  changed={k:[original[k],c[k]] for k in c if original.get(k)!=c[k]};assert set(changed)=={'feature_dir','buffer_dir','experiment_dir','run_root','contract'}
  cfgreview[case][arm]=changed;put(d/('resolved_'+arm+'.json'),c)
 source={p.relative_to(d/'source').as_posix():sha(p) for p in sorted((d/'source').rglob('*')) if p.is_file()}
 modal=json.loads(parents['MODAL_CONTRACT.json']['text']);modal['source_sha256']=source;modal['parent_modal_sha256']=parents['MODAL_CONTRACT.json']['sha256'];modal['variant']='Random_MLR_IFA_plus_ALIGN_only'
 put(d/'MODAL_CONTRACT.json',modal)
 files=dict(x['files']);files.update(json.loads(parents['CONTRACT.json']['text'])['inputs'])
 for p in d.rglob('*.json'):
  if p.name!='CONTRACT.json':files[dest+'/'+p.relative_to(d).as_posix()]=sha(p)
 contract=dict(case=case,source_root=Q+'/cases/'+case+'/source',source_sha256=source,files=files,production_optimizer_calls=x['production_budget'],admission_optimizer_calls=20,epochs=400,formal_episodes=200,training_environments=0)
 put(d/'CONTRACT.json',contract)
 summary[case]=dict(production_budget=x['production_budget'],stages={t:{k:m[k] for k in ['population','batches','bc_valid']} for t,m in x['manifests'].items()},labels=x['labels'],source_files=len(source),input_files=len(x['files']),contract_sha256=sha(d/'CONTRACT.json'))
put(R/'CONFIGURATION_REVIEW.json',dict(status='PASS_PATH_ONLY_CONFIG_CHANGES',cases=cfgreview))
put(R/'INPUT_AUDIT.json',dict(status=e['status'],observed=e['observed'],actual_evidence_sha256=sha(R/'capacity_readonly_20261109_164643/EVIDENCE.json'),cases=summary,new_gpu_calls=0,new_optimizer=0,new_environments=0))
put(R/'ACTIVE_ROOT.json',dict(status='PREPARED_NOT_DEPLOYED',remote_root=Q,private_root=P,current_q1_root='/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1',current_q1_contract='fbbce2917380d1cde1025f0006b6fdd6bc7959f92226bb5588cc431ee0c8d96c'))
print('Prepared six path-only configs, 24 actual Random manifests and input contracts')
