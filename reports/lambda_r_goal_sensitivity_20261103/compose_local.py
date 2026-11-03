"""Prepare the authorized fixed lambda grid; never launches a science process."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent;P=R.parent
Q=P/'20261031_full_q_vs_latest_s_two_suites';S=P/'20261029_support_gate_full_s_goal'
NAME='S_LAMBDA_R_GOAL41_42_43_20261103_v1'
CASES=[g+'_'+str(k) for g in ['L005','L020'] for k in (41,42,43)]
read=lambda p:p.read_text(encoding='utf8')
js=lambda p:json.loads(read(p))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
def dump(p,x):put(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists(),'No rebuilding frozen bundle'
binding=js(Q/'S_ACCEPTANCE.json');binding['sources']={'GOAL':binding['sources']['GOAL']};binding['cases']={k:v for k,v in binding['cases'].items() if k.startswith('GOAL_')}
baseline=Path(binding['sources']['GOAL']['final_path'])
assert sha(baseline/'EVIDENCE.json')==binding['sources']['GOAL']['evidence_sha256']
assert sha(baseline/'LOCAL_FINAL_AUDIT.json')==binding['sources']['GOAL']['audit_sha256']
dump(R/'S_ACCEPTANCE.json',binding)
c=read(Q/'controller.py').replace('FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1',NAME)
c=c.replace("['GOAL','OBJECT']","['L005','L020']").replace("suite=='OBJECT'","suite=='L020'").replace("'GOAL_EVALUATION_COMPLETE.json'","'L005_EVALUATION_COMPLETE.json'")
c=c.replace('train_Q','train_S').replace("str(root/'train.py'),'Q'","str(root/'train.py'),'S'")
c=c.replace('Unique finite Q dispatch','Unique finite S lambda-grid dispatch')
put(R/'controller.py',c)
review={}
for case in CASES:
 group,seed=case.split('_');seed=int(seed);lam=.05 if group=='L005' else .2;sr=S/('seed%d'%seed)
 names=['runtime.py','updates.py','legacy_updates.py','memory.py','legacy_memory.py','support_memory.py','losses.py','legacy_losses.py','synthetic_checks.py','fork_tools.py','gate_admission.py','initialization.py','collect.py','evaluate.py','train.py','layout.py']
 mapping={n:read(sr/n) for n in names}
 layout=mapping['layout.py'].replace('SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1',NAME)
 layout=rep(layout,"DATA_PRIVATE/('seed%d'%SEED)/a",'DATA_PRIVATE/'+repr(case)+'/a')
 layout=rep(layout,'RETAIN_WEIGHT=.1','RETAIN_WEIGHT='+repr(lam))
 layout=layout.replace("'/root':0","'/root':128*1024**2").replace("'/data':8*1024**3","'/data':12*1024**3")
 mapping['layout.py']=layout+'\nSUITE="GOAL"\nLAMBDA_GROUP='+repr(group)+'\n'
 mapping['runtime.py']=read(Q/'engineering_startup_v2'/('GOAL_'+str(seed))/'runtime.py').replace('SCHEDULE_Q_','SCHEDULE_S_')
 mapping['candidate.py']=read(sr/'candidate.py')
 mapping['initialization.py']=read(Q/'engineering_startup_v2'/('GOAL_'+str(seed))/'initialization.py')
 mapping['gate_admission.py']=rep(mapping['gate_admission.py'],"0. if arm=='C' else .1","0. if arm=='C' else RETAIN_WEIGHT")
 mapping['fork_tools.py']=rep(mapping['fork_tools.py'],'re=.1*(rx+.1*rt)','re=RETAIN_WEIGHT*(rx+.1*rt)')
 t=mapping['train.py']
 t=rep(t,'            if stage==6:\n                admit(',"            set_count_mode('admission');admission_before=CALLS['admission']\n            if stage==6:\n                admit(")
 t=rep(t,'            for ep in range(100):',"            assert CALLS['admission']-admission_before==(4 if stage==6 else 29)\n            set_count_mode('production')\n            for ep in range(100):")
 t=rep(t,'                assert bi==len(loader)',"                assert bi==len(loader) and CALLS['production']==physical")
 t=t.replace('physical_optimizer_calls=physical,gate_blocks=',"physical_optimizer_calls=physical,physical_calls_observed=dict(CALLS),gate_blocks=")
 t=t.replace("method='support_gate_full_S'","method='support_gate_full_S',lambda_R=RETAIN_WEIGHT,uniform_reference='same_lambda_uniform_support'")
 t=t.replace("status='PASS_FROZEN_INPUT',manifest_sha256=","status='PASS_FROZEN_INPUT',lambda_R=RETAIN_WEIGHT,manifest_sha256=")
 needle="                write(R/('TASK6_PARITY_'+arm+'.json'),dict("
 t=rep(t,needle,"                sr=js(R/'S_REFERENCE.json')['stage6_checkpoint'];assert sha(sr['path'])==sr['sha256']\n                assert equal(actual,torch.load(sr['path'],map_location='cpu')['state_dict']),'Task6 accepted S tensor mismatch'\n"+needle)
 t=t.replace("same_seed_new_arm_comparator='NOT_RUN_SINGLE_S'","same_seed_new_arm_comparator='ACCEPTED_S_ALL_TENSORS_EXACT',accepted_S_cp_sha256=sr['sha256']")
 t=rep(t,'        assert a.step==76100',"        assert CALLS=={'production':142200,'admission':91}\n        assert a.step==76100")
 mapping['train.py']=t
 unchanged=['updates.py','legacy_updates.py','memory.py','legacy_memory.py','support_memory.py','losses.py','legacy_losses.py','synthetic_checks.py','collect.py','evaluate.py']
 for n in unchanged:assert mapping[n]==read(sr/n),n
 for n,s in mapping.items():
  ast.parse(s,feature_version=(3,8));original=read(sr/n) if (sr/n).exists() else read(sr/'candidate.py')
  review[case+'/'+n]=dict(source=str(sr/n),original_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if s!=original:put(R/'review'/(case+'_'+n+'.diff'),''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(sr/n),tofile=case+'/'+n)))
  put(R/case/n,s)
 parent=js(sr/'PARENT_BINDING.json');parent.update(suite='GOAL',seed=seed,lambda_R=lam,lambda_group=group)
 dump(R/case/'PARENT_BINDING.json',parent);put(R/case/'INITIAL_SPLIT.json',read(sr/'INITIAL_SPLIT.json'))
 dump(R/case/'S_REFERENCE.json',binding['cases']['GOAL_'+str(seed)]['source_S_reference'])
dump(R/'SOURCE_REVIEW.json',dict(status='STATIC_SOURCE_REVIEW_NOT_GPU_ACCEPTANCE',science_change='Only lambda_R=0.05 or 0.2; same gate, support, sampling, losses otherwise',files=review,per_stage_actual_admission='REQUIRED'))
d=read(Q/'deploy_remote.py').replace('FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1',NAME)
d=d.replace("['GOAL','OBJECT']","['L005','L020']").replace("('/root',0,1024**3,False)","('/root',128*1024**2,1024**3,False)").replace("('/data',8*1024**3","('/data',12*1024**3")
d=rep(d,"assert (old/'engineering_suite_order_20261030_v1/controller.exit').read_text()=='0'","assert (old/'engineering_suite_order_20261030_v1/controller.exit').read_text()=='0'\ncompleted_q=Path('/data/mlr_ifa/experiment_workspaces/FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2')\nassert (completed_q/'controller.exit').read_text().strip()=='0'")
d=d.replace('[str(Q),str(old),','[str(Q),str(old),str(completed_q),')
d=d.replace("method='Q_support_weighted_no_gate'","method='S_support_gate',lambda_R=parent['lambda_R'],lambda_group=parent['lambda_group']")
d=d.replace('physical_optimizer_calls=total,C_candidate_calls=0','physical_optimizer_calls=142200,C_candidate_calls=66100')
d=d.replace('accepted_updates=544800,physical_optimizer_calls=544800,C_candidate_calls=0','accepted_updates=456600,physical_optimizer_calls=853200,C_candidate_calls=396600')
put(R/'deploy_remote.py',d)
check=read(Q/'check_readonly.py').replace('FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2',NAME).replace("['GOAL','OBJECT']","['L005','L020']").replace("('/root',0,1024**3,False)","('/root',128*1024**2,1024**3,False)").replace("('/data',8*1024**3","('/data',12*1024**3")
check=check.replace("if mount=='/data' else 0","if mount=='/data' else bound")
put(R/'check_readonly.py',check)
pub=read(Q/'publish.py').replace('full_q_vs_latest_s_two_suites_20261031','lambda_r_goal_sensitivity_20261103').replace('Full Q versus accepted S two suites','GOAL S lambda_R fixed-grid sensitivity')
put(R/'publish.py',pub)
for p in R.rglob('*.py'):ast.parse(read(p),feature_version=(3,8))
print(json.dumps(dict(status='COMPOSED_NOT_FROZEN_OR_DISPATCHED',cases=CASES)))
