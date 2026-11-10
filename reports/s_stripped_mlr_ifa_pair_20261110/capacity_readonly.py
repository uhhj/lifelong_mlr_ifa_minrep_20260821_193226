import json,hashlib,os,time,subprocess
from pathlib import Path
PREV=Path('/data/mlr_ifa/experiments/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
J=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  name=suite+'_'+str(seed); inp=J(PREV/name/'INPUT.json')
  lab=Path(inp['labels']['6']['path']); old=lab.parents[3]
  # bindings are below OLD/effects/current/task6, so parents[3] is OLD.
  source_contract=J(old/'CONTRACT.json')
  source_root=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source') if name=='OBJECT_41' else Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022')/name/'source'
  actual={str(source_root/k):sha(source_root/k) for k in source_contract['source_sha256']}
  assert all(actual[str(source_root/k)]==h for k,h in source_contract['source_sha256'].items())
  result[name]=dict(align_parent=str(old),source_root=str(source_root),source_hashes=actual,eval_config=J(old/'resolved_eval.json'),ra_config=J(old/'resolved_ra_coreset.json'),source_contract_sha256=sha(old/'CONTRACT.json'),build_source=(source_root/'caet_v3_exp/train.py').read_text(),model_source=(source_root/'caet_v3_exp/model.py').read_text())
print(json.dumps(dict(status='PASS_READONLY_S_PARENT_SOURCE',observed=time.time(),cases=result,new_optimizer=0,new_environment=0)))
