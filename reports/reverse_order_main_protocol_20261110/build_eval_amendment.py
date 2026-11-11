"""Prepare the explicitly requested common DEBR evaluator, without training."""
from pathlib import Path
import ast,copy,hashlib,json,difflib,datetime
R=Path(__file__).resolve().parent
A=R/'evaluation_unification_v1'
A.mkdir(exist_ok=True)
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda b:hashlib.sha256(b).hexdigest()
def put(p,t):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8',newline='\n')
def dump(p,x):put(p,json.dumps(x,ensure_ascii=True,indent=2)+'\n')
old=J(R/'CONTRACT.json');parent=sha((R/'CONTRACT.json').read_bytes())
assert parent=='5d28905059a9afc15cc8175097eee4ce3af5599154bb980bc54dce53d220795c'
for f in ['CONTRACT.json','run_case.py','final_cpu_case.py','check_readonly.py','PROTOCOL.md','IMPLEMENTATION_NOTES.md']:
    p=A/'original'/f
    if p.exists():assert p.read_bytes()==(R/f).read_bytes()
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((R/f).read_bytes())

source=R/'cases/OBJECT_DEBR_41/evaluate.py'
t=source.read_text(encoding='utf-8-sig')
assert t.count('from runtime import *')==1 and t.count("p=load('S',6,c,shape)")==1
t=t.replace('from runtime import *','from unified_mlr_runtime import *').replace("p=load('S',6,c,shape)","p=load('MLR',6,c,shape)")
put(A/'evaluate.py',t)

runtime='''"""Evaluation-only adapter; no training methods, optimizer or acquisition."""
from pathlib import Path
import os,sys,json,hashlib,time,copy
import numpy as np
import torch
from easydict import EasyDict
Q=Path(__file__).resolve().parent.parent
R=Path(os.environ['REVERSE_CASE_ROOT'])
INPUT=json.loads((R/'INPUT.json').read_text())
assert INPUT['method']=='MLR'
O=Path(INPUT['input_root']);SEED=INPUT['seed'];ARMS=['MLR']
from order import install,ORIGINAL_TASKS
install()
from coreset_exp.common import sha,write
from libero.lifelong.utils import control_seed
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
def js(p):return json.loads(Path(p).read_text())
def model_path(arm,stage):
 assert arm=='MLR' and stage in range(6,10)
 return Path(INPUT['output_root'])/'models'/('mlrifa_%d_model_ep99.pth'%stage)
def verify():
 from reverse_common import verify_case
 m=js(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 c=js(Q/'CONTRACT.json')
 assert m['id']=='COMMON_DEBR_EVALUATION_20261111_V1'
 assert c['evaluation_amendment']['manifest_sha256']==sha(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 assert m['cases'][R.name]['training_contract_sha256']==sha(R/'CONTRACT.json')
 for name,want in m['evaluation_source_sha256'].items():assert sha(Q/name)==want,name
 return verify_case(R)
def capacity():
 from reverse_common import capacity as check
 return check()
'''
put(A/'unified_mlr_runtime.py',runtime)

entry='''"""Run the common full-DEBR rollout on already-trained pure MLR weights."""
from pathlib import Path
import json,os,runpy,sys,time
def main():
 Q=Path(__file__).resolve().parent.parent;A=Q/'evaluation_unification_v1';R=Path(os.environ['REVERSE_CASE_ROOT'])
 sys.path.insert(0,str(A))
 import unified_mlr_runtime as rt
 rt.verify();inp=rt.INPUT
 assert (R/'train.exit').read_text().strip()=='0'
 native=Path(inp['output_root'])/'evaluation'
 assert not list(native.glob('trials*.jsonl')) and not list(native.glob('attempts*.jsonl')),'Old native evaluation must not already have run'
 assert not (R/'EVALUATION_STARTED.json').exists() and not (R/'NORMALIZED_OUTCOMES.json').exists()
 manifest=rt.js(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 rt.write(R/'EVALUATION_PROTOCOL_DECLARATION.json',dict(status='DECLARED_BEFORE_ANY_FORMAL_EPISODE',time=time.time(),protocol=manifest['id'],method='MLR',training_contract_sha256=rt.sha(R/'CONTRACT.json'),manifest_sha256=rt.sha(Q/'EVALUATION_PROTOCOL_AMENDMENT.json'),source_sha256=manifest['evaluation_source_sha256'],extra_training_updates=0))
 runpy.run_path(str(A/'evaluate.py'),run_name='__main__')
 raw=rt.js(R/'OUTCOMES.json');assert len(raw)==200
 digest=rt.sha(R/'OUTCOMES.json')
 records=[dict(x,method='MLR',evaluation_protocol=manifest['id'],source_outcomes_sha256=digest) for x in raw]
 rt.write(R/'NORMALIZED_OUTCOMES.json',records)
 rt.write(R/'EVALUATION_PROTOCOL_COMPLETE.json',dict(status='PASS_COMMON_DEBR_EVALUATION_COMPLETE_PENDING_FINAL_AUDIT',protocol=manifest['id'],episodes=200,normalised_sha256=rt.sha(R/'NORMALIZED_OUTCOMES.json'),source_outcomes_sha256=digest,training_repeated=False,time=time.time()))
'''
put(A/'unified_mlr_entry.py',entry)

run=(R/'run_case.py').read_text(encoding='utf-8-sig')
start=run.index('        from coreset_exp import evaluate\n')
end=run.index('\nelse:\n    from easydict',start)
amended=run[:start]+"        sys.path.insert(0,str(Q/'evaluation_unification_v1'))\n        from unified_mlr_entry import main as evaluate_common\n        evaluate_common()"+run[end:]
put(A/'run_case_amended.py',amended)
# Only the MLR evaluation branch may change; no training branch is rewritten.
def phase_branches(s):
 tree=ast.parse(s)
 top=next(x for x in tree.body if isinstance(x,ast.If) and ast.unparse(x.test)=="inp['method'] == 'MLR'" and any(isinstance(y,ast.If) and ast.unparse(y.test)=="args.phase == 'train'" for y in x.body))
 mlr=next(x for x in top.body if isinstance(x,ast.If) and ast.unparse(x.test)=="args.phase == 'train'")
 return ast.dump(ast.Module(body=mlr.body,type_ignores=[])),ast.dump(ast.Module(body=top.orelse,type_ignores=[]))
assert phase_branches(run)==phase_branches(amended)

cpu=(R/'final_cpu_case.py').read_text(encoding='utf-8-sig')
cpu=cpu.replace('self.module=BriefPolicy(copy.deepcopy(cfg),None)',"shape=cfg.get('shape_meta') or inp['eval_config'].get('shape_meta') or J(Q/'cases'/('%s_MLR_%d'%(inp['suite'],inp['seed']))/'INPUT.json')['eval_config']['shape_meta']\n        self.module=BriefPolicy(copy.deepcopy(cfg),copy.deepcopy(shape))")
start=cpu.index("    if method=='MLR':\n        base=out/'evaluation';native=")
end=cpu.index('\ncache_checks={}',start)
newblock='''    eps=np.random.RandomState(20261024+inp['seed']*1000+t*100+i).standard_normal((600,7)).astype(np.float32)
    assert x['noise_sha256']==ah(eps) and x['env_seed']==inp['seed']+i//10
    assert 1<=x['steps']<=600 and x['termination']==('success' if x['success'] else 'timeout600')
    assert x['success'] or x['steps']==600
    if method=='MLR':
        manifest=J(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
        assert x['evaluation_protocol']==manifest['id']
        assert x['source_outcomes_sha256']==sha(R/'OUTCOMES.json')
        raw_records=J(R/'OUTCOMES.json')
        raw=next(v for v in raw_records if (v['stage'],v['task'],v['initial_index'])==(s,t,i))
        assert all(x[k]==v for k,v in raw.items())
        declared=J(R/'EVALUATION_PROTOCOL_DECLARATION.json')
        assert declared['manifest_sha256']==sha(Q/'EVALUATION_PROTOCOL_AMENDMENT.json') and declared['training_contract_sha256']==contract
        assert J(R/'EVALUATION_PROTOCOL_COMPLETE.json')['episodes']==200
'''
cpu=cpu[:start]+newblock+cpu[end:]
cpu=cpu.replace("noise_definition='native_saved_global_RNG' if method=='MLR' else 'native_per_trajectory_Normal'","noise_definition='common_full_DEBR_per_trajectory_Normal',evaluation_protocol='COMMON_DEBR_EVALUATION_20261111_V1'")
put(A/'final_cpu_case_amended.py',cpu)

check=(R/'check_readonly.py').read_text(encoding='utf-8-sig')
assert "'FINAL_CPU_REQUEST.json']" in check
check=check.replace("'FINAL_CPU_REQUEST.json']","'FINAL_CPU_REQUEST.json','EVALUATION_PROTOCOL_AMENDMENT.json','EVALUATION_PROTOCOL_AMENDMENT_APPLIED.json']")
put(A/'check_readonly_amended.py',check)

cases={};diffs={}
def flatten(x,p=''):
 if isinstance(x,dict):
  out={}
  for k,v in x.items():out.update(flatten(v,p+k+'.'))
  return out
 return {p[:-1]:x}
allowed={'buffer_dir','contract','experiment_dir','feature_dir','run_root','selector','shape_meta'}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  name='%s_MLR_%d'%(suite,seed);peer='%s_DEBR_%d'%(suite,seed)
  a=J(R/'cases'/name/'INPUT.json');b=J(R/'cases'/peer/'INPUT.json')
  aa={k:v for k,v in a['eval_config'].items() if k not in allowed};bb={k:v for k,v in b['eval_config'].items() if k not in allowed}
  assert aa==bb,(name,'scientific eval configuration differs')
  # Ensure the root entry-point change cannot invalidate an active case contract.
  cc=J(R/'cases'/name/'CONTRACT.json')
  assert not any(Path(p).name in ['run_case.py','final_cpu_case.py','check_readonly.py'] and '/cases/' not in p for p in cc['files'])
  cases[name]=dict(peer_DEBR_case=peer,training_contract_sha256=sha((R/'cases'/name/'CONTRACT.json').read_bytes()),input_sha256=sha((R/'cases'/name/'INPUT.json').read_bytes()),scientific_eval_config_equal=True,metadata_difference_keys=[k for k in a['eval_config'] if a['eval_config'].get(k)!=b['eval_config'].get(k)],model_paths={str(s):a['output_root']+'/models/mlrifa_%d_model_ep99.pth'%s for s in range(6,10)})
  diffs[name]=cases[name]['metadata_difference_keys']

files={
 'evaluation_unification_v1/evaluate.py':(A/'evaluate.py').read_text(),
 'evaluation_unification_v1/unified_mlr_runtime.py':runtime,
 'evaluation_unification_v1/unified_mlr_entry.py':entry,
 'evaluation_unification_v1/probe_case.py':(A/'probe_case.py').read_text(encoding='utf-8'),
 'evaluation_unification_v1/original/CONTRACT.json':(A/'original/CONTRACT.json').read_text(),
 'evaluation_unification_v1/original/run_case.py':run,
 'evaluation_unification_v1/original/final_cpu_case.py':(A/'original/final_cpu_case.py').read_text(),
 'evaluation_unification_v1/original/check_readonly.py':(A/'original/check_readonly.py').read_text(),
}
manifest=dict(id='COMMON_DEBR_EVALUATION_20261111_V1',status='USER_AUTHORIZED_EVALUATION_PROTOCOL_AMENDMENT',user_request='MLR+IFA同样采用完整方案协议',parent_root_contract_sha256=parent,scope='Six current reverse-order MLR cases: full-DEBR evaluator, pure original MLR training unchanged; no completed training repeated; DEBR training/evaluation unchanged.',cases=cases,evaluation_source_sha256={p:sha(s.encode()) for p,s in files.items() if '/original/' not in p},template_parent=str(source.relative_to(R)).replace('\\','/'),template_parent_sha256=sha(source.read_bytes()),permitted_template_changes=['runtime import module','arm label only'],common_settings=dict(episodes=200,initial_indices='0..19',max_steps=600,zero_steps=5,lanes=10,groups=2,history=8,normal='per-trajectory np.RandomState(20261024+seed*1000+task*100+initial)',success='original environment signal latched',done_action='zero',environment_lifecycle='new per group',cudnn_benchmark=False,cache_parity_atol=1e-4,cache_parity_rtol=1e-4),training_epochs_per_task=100,training_methods_preserved=True,training_case_contracts_unchanged=True,formal_not_started_required=True,extra_training_updates=0,extra_formal_episodes=0,original_order_MLR_reference='Historical native evaluator; not re-evaluated or replaced by this amendment')
dump(A/'AMENDMENT.json',manifest)
files['EVALUATION_PROTOCOL_AMENDMENT.json']=(A/'AMENDMENT.json').read_text()
files.update({'run_case.py':amended,'final_cpu_case.py':cpu,'check_readonly.py':check})
new=copy.deepcopy(old)
for p,t in files.items():new['files'][p]=sha(t.encode())
new['evaluation_amendment']=dict(id=manifest['id'],parent_contract_sha256=parent,manifest='EVALUATION_PROTOCOL_AMENDMENT.json',manifest_sha256=sha(files['EVALUATION_PROTOCOL_AMENDMENT.json'].encode()),user_request=manifest['user_request'],training_case_contracts_unchanged=True)
dump(A/'CONTRACT_amended.json',new)
newtext=(A/'CONTRACT_amended.json').read_text()
proof=dict(status='STATIC_PREPARATION_NOT_REMOTE_APPLIED',parent_contract_sha256=parent,new_contract_sha256=sha(newtext.encode()),all_12_training_case_contracts_identical_to_parent=all(old['cases'][k]==new['cases'][k] for k in old['cases']),MLR_training_branch_AST_equal=True,entire_DEBR_branch_AST_equal=True,eval_config_differences_metadata_only=diffs,template_changes_only=manifest['permitted_template_changes'],payloads={p:dict(sha256=sha(t.encode()),bytes=len(t.encode())) for p,t in files.items()},no_training_or_environments=True)
dump(A/'STATIC_REVIEW.json',proof)
dump(A/'REVIEWED_TEXT_FILES.json',dict(files=files,contract_text=newtext,parent_contract_sha256=parent,new_contract_sha256=proof['new_contract_sha256']))
for f in ['run_case.py','final_cpu_case.py','check_readonly.py']:
 dump(A/(f+'.diff.json'),dict(diff=''.join(difflib.unified_diff((R/f).read_text().splitlines(True),files[f].splitlines(True),fromfile='original/'+f,tofile='amended/'+f))))
print(json.dumps(dict(status=proof['status'],new_contract_sha256=proof['new_contract_sha256'],files=len(files),training_changes=0,formal_added=0)))
