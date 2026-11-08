"""Prepare a fresh, finite q=1 GOAL ablation. Never launch or modify old runs."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent;W=R.parents[1];E=R.parent
G=E/'20261103_lambda_r_goal_sensitivity';O=G
NAME='S_Q1_GOAL41_42_43_20261108_v1';CASES=['Q1_'+str(s) for s in (41,42,43)]
def read(p):return p.read_text(encoding='utf-8-sig')
def js(p):return json.loads(read(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8',newline='\n')
def dump(p,v):put(p,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
def amounts(s):
 for a,b in [('S_LAMBDA_R_GOAL41_42_43_20261103_v1',NAME),('76100','76100'),('142200','142200'),('6700','6700')]:s=s.replace(a,b)
 return s
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists(),'Already frozen: no rebuild'
bind=js(O/'S_ACCEPTANCE.json');bind['sources']['GOAL']['final_path']=str((E/'20261029_support_gate_full_s_goal/final_readonly_20261030_111546').resolve());bind['sources']={'GOAL':bind['sources']['GOAL']};bind['cases']={k:v for k,v in bind['cases'].items() if k.startswith('GOAL_')}
for n,k in [('EVIDENCE.json','evidence_sha256'),('LOCAL_FINAL_AUDIT.json','audit_sha256')]:assert sha(Path(bind['sources']['GOAL']['final_path'])/n)==bind['sources']['GOAL'][k]
dump(R/'S_ACCEPTANCE.json',bind)
review={}
for case in CASES:
 seed=int(case.split('_')[1]);src=O/('L005_'+str(seed));base={p.name:read(p) for p in src.glob('*.py')}
 layout=base['layout.py'].replace('S_LAMBDA_R_GOAL41_42_43_20261103_v1',NAME).replace('L005_'+str(seed),case).replace('RETAIN_WEIGHT=0.05','RETAIN_WEIGHT=0.1').replace('12*1024**3','4*1024**3')
 layout=layout.replace("LAMBDA_GROUP='L005'","LAMBDA_GROUP='Q1'\nSUPPORT_CALIBRATION='DISABLED_FIXED_ONE'")
 base['layout.py']=layout
 base['train.py']=amounts(read(G/('L005_'+str(seed))/'train.py'))
 base['initialization.py']=read(G/('L005_'+str(seed))/'initialization.py').replace('lambda_only_hardware_controlled','q_only_hardware_controlled')
 sm=base['support_memory.py']
 sm=rep(sm,"if arm in ['Q','G']:return q", "if arm in ['Q','G']:return {t:(1. if memory[t]['rows'] else 0.) for t in q}")
 base['support_memory.py']=sm
 t=base['train.py']
 t=rep(t,"support=memory[t]['support'],support_trajectories=", "support=memory[t]['support'],effective_support=(1. if memory[t]['rows'] else 0.),support_trajectories=")
 base['train.py']=t
 ft=base['fork_tools.py']
 ft=rep(ft,"support=weights(memory,'Q'),", "support=weights(memory,'Q'),empirical_support={str(t):m['support'] for t,m in memory.items()},support_mode='FIXED_ONE_NONEMPTY',")
 base['fork_tools.py']=ft
 ga=base['gate_admission.py']
 ga=rep(ga,"first=step(a,b,'Q' if arm=='G' else arm,m,scale,info);one=capture(a);calls+=1", "first=step(a,b,'Q' if arm=='G' else arm,m,scale,info);one=capture(a);calls+=1\n    assert equal(one,ss),'Q1 actual-memory update must equal uniform legacy retention'")
 ga=rep(ga,"assert torch.allclose(rx,.1*sx,atol=1e-7,rtol=1e-6) and torch.allclose(rt,.1*st,atol=1e-7,rtol=1e-6)","assert torch.equal(rx,sx) and torch.equal(rt,st)")
 ga=ga.replace('q_point_one_preserved=True','raw_q_point_one_ignored=True,effective_q_one_exact=True')
 ga=rep(ga,"probe_all_state_exact=True,", "probe_all_state_exact=True,q1_actual_memory_uniform_exact=True,q1_short_history_exact=True,")
 base['gate_admission.py']=ga
 parent=js(src/'PARENT_BINDING.json');parent.update(lambda_R=.1,lambda_group='Q1',support_calibration='DISABLED_FIXED_ONE',seed=seed,suite='GOAL')
 for name,s in base.items():
  ast.parse(s,feature_version=(3,8));put(R/case/name,s)
  old=read(src/name)
  review[case+'/'+name]=dict(source=str(src/name),source_sha256=sha(src/name),new_text_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==old)
  if s!=old:put(R/'review'/(case+'_'+name+'.diff'),''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(src/name),tofile=case+'/'+name)))
 dump(R/case/'PARENT_BINDING.json',parent);put(R/case/'INITIAL_SPLIT.json',read(src/'INITIAL_SPLIT.json'));dump(R/case/'S_REFERENCE.json',bind['cases']['GOAL_'+str(seed)]['source_S_reference'])
wrapper=read(G/'train_wrapper.py')
wrapper=rep(wrapper,"[g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]",repr(CASES))
a=wrapper.index("validation=rt.js(ROOT/'ENGINEERING_VALIDATION.json')");b=wrapper.index("assert os.environ['WORLD_SIZE']",a)
wrapper=wrapper[:a]+"reuse=rt.js(ROOT/'ENGINEERING_REUSE.json');assert reuse['status']=='REVIEWED_REUSE_WITH_OWN_STAGE_EQUALITY_REQUIRED'\n"+wrapper[b:]
put(R/'train_wrapper.py',wrapper)
env=read(G/'environment_locked.py').replace('S_LAMBDA_R_GOAL41_42_43_20261103_v1',NAME)
env=rep(env,"[g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]",repr(CASES));put(R/'environment_locked.py',env)
check=read(G/'check_readonly.py').replace('S_LAMBDA_R_GOAL41_42_43_20261103_v1',NAME).replace('12*1024**3','4*1024**3')
check=rep(check,"[s+'_'+str(k) for s in ['L005','L020'] for k in [41,42,43]]",repr(CASES));put(R/'check_readonly.py',check)
audit=read(G/'audit_progress.py').replace("['L005_41', 'L005_42', 'L005_43', 'L020_41', 'L020_42', 'L020_43']",repr(CASES)).replace("suite='GOAL'","suite='GOAL'")
audit=rep(audit,"q==len(ids)/10", "q==(1. if ids else 0.) and pr['empirical_support'][t]==len(ids)/10")
audit=rep(audit,"pr=js(prefix+'PROBES_%d.json'%st)","assert a['q1_actual_memory_uniform_exact'] and a['q1_short_history_exact']\n    pr=js(prefix+'PROBES_%d.json'%st);assert pr['support_mode']=='FIXED_ONE_NONEMPTY'")
put(R/'audit_progress.py',audit);put(R/'audit_seed_evaluation.py',read(G/'audit_seed_evaluation.py'))
scripts={}
for case in CASES:
 seed=int(case.split('_')[1]);p=js(R/case/'PARENT_BINDING.json')
 s=amounts(read(G/'final_L005_41_readonly.py'))
 lines=s.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith('Q=Path('));lines[i]="Q=Path('/data/mlr_ifa/experiment_workspaces/"+NAME+'/'+case+"');O=Path("+repr(p['output'])+");S=Path("+repr(p['source'])+")";s='\n'.join(lines)+'\n'
 s=rep(s,"pr['support'][str(t)]==len(trajectories)/10.","pr['support'][str(t)]==(1. if trajectories else 0.) and pr['empirical_support'][str(t)]==len(trajectories)/10.")
 s=rep(s,"assert binding['support']==len(trajectories)/10.","assert binding['effective_support']==(1. if trajectories else 0.) and binding['support']==len(trajectories)/10.")
 put(R/('final_'+case+'_readonly.py'),s);scripts[case]=s
aggregator=read(G/'final_readonly.py');aggregator=aggregator[aggregator.index('"""Only after natural completion:'):]
put(R/'final_readonly.py','CHECK_SCRIPT='+repr(check)+'\nSCRIPTS='+repr(scripts)+'\n'+aggregator)
af=amounts(read(G/'audit_final_local.py')).replace("['L005_41', 'L005_42', 'L005_43', 'L020_41', 'L020_42', 'L020_43']",repr(CASES))
for a,b in [('2400','1200'),('456600','228300'),('853200','426600'),('==546','==273'),('==246','==123'),('==240','==120'),('==1200','==600'),("'collect':24,'evaluate':6","'collect':12,'evaluate':3"),('epochs=2400','epochs=1200'),('isolated_optimizer_updates=546','isolated_optimizer_updates=273'),('formal_extra_autograd=216','formal_extra_autograd=108'),('admission_extra_autograd=246','admission_extra_autograd=123'),('training_collection_episodes=240','training_collection_episodes=120'),('formal_episodes=1200','formal_episodes=600'),('vectors=120,cells=60','vectors=60,cells=30')]:af=af.replace(a,b)
# Global numeric substitutions above are audited below, never used as acceptance claims.
af=af.replace("a['epochs']==600","a['epochs']==1200").replace('epochs=1200,main_updates','epochs=1200,main_updates')
af=af.replace('Only fixed lambda_R 0.05/0.2 GOAL S41/42/43; accepted 0.1 read-only. No outcome-based selection.','Only GOAL S q=1 seed41/42/43; lambda_R=0.1; accepted calibrated S read-only. No outcome-based selection.').replace('Fresh GOAL3090','Fresh GOAL3090').replace('lambda-only','q-only').replace('GOAL engineering validation costs separate from production.','Prior engineering evidence reused; each stage own admission required.')
put(R/'audit_final_local.py',af)
comp=read(G/'compare_primary_s.py')
for a,b in [("['L005','L020']","['Q1']"),("['GOAL']","['GOAL']"),('1200','600'),('456600','228300'),('==546','==273'),('GOAL两个新系数各三seed','GOAL q=1三个seed'),('固定系数敏感性','固定q=1消融'),('两个新系数共240次','本次三个seed共120次'),('每档各120次','本次120次'),('Baseline 0.1 historical diagnostics remain separate; the baseline 600 outcomes are reused for two paired comparisons, not 600 independent baseline outcomes','Baseline historical diagnostics remain separate; 600 outcomes are reused once in a paired comparison'),('lambda-only','q-only')]:comp=comp.replace(a,b)
put(R/'compare_primary_s.py',comp)
dump(R/'SOURCE_REVIEW.json',dict(status='STATIC_REVIEW_PREDEPLOY',science_change='Only effective support q_j=1 for nonempty cache, zero contribution for empty cache, lambda_R=0.1; candidate loss, old probe, directional diagnostic consistent',files=review))
dump(R/'ENGINEERING_REUSE.json',dict(status='REVIEWED_REUSE_WITH_OWN_STAGE_EQUALITY_REQUIRED',wrapper_source=str(G/'train_wrapper.py'),wrapper_source_sha256=sha(G/'train_wrapper.py'),object_evidence=str(G/'ENGINEERING_ACCEPTANCE.json'),object_evidence_sha256=sha(G/'ENGINEERING_ACCEPTANCE.json'),note='No new full-path GPU validation claimed. Same generic device clone/check implementation; every new stage has actual CPU/device capture parity and original predicate parity plus its own GPU admission. Q1 GPU admission asserts actual-memory Q update equals original uniform retention; fake q=.1 is ignored exactly.',new_engineering_optimizer_calls=0,new_engineering_environments=0))
for p in R.rglob('*.py'):ast.parse(read(p),feature_version=(3,8))
dump(R/'FINAL_WRAPPER_REVIEW.json',dict(status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE',required_actual=dict(checkpoints=12,recoveries=3,caches=12,initial_noise_episodes=600,gate_blocks=20100,production_calls=426600),scale_exceptions='NONE; any observed mismatch must be investigated and preserved, not pre-waived',files={p.name:sha(p) for p in R.glob('final*.py')}))
print(json.dumps(dict(status='PREPARED_NOT_FROZEN_NOT_DEPLOYED',cases=CASES)))
