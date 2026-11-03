"""Prepare independent GOAL audit wrappers; execution remains required for acceptance."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent;B=R.parent/'20261101_lambda_r_object_sensitivity'
read=lambda p:p.read_text(encoding='utf8')
def put(n,s):ast.parse(s,feature_version=(3,8));(R/n).write_text(s,encoding='utf8')
def amounts(s):
 for a,b in [('S_LAMBDA_R_OBJECT41_42_43_20261101_v1','S_LAMBDA_R_GOAL41_42_43_20261103_v1'),('633000','456600'),('1171800','853200'),('105500','76100'),('195300','142200'),('9100','6700'),('54600','40200')]:s=s.replace(a,b)
 return s
a=amounts(read(B/'audit_progress.py')).replace("suite='OBJECT'","suite='GOAL'")
a=a.replace("['tensors','rng','hardware']","['tensors','rng']")
a=a.replace("=='ACCEPTED_S_ALL_TENSORS_EXACT'","=='AUTHORIZED_CROSS_GPU_COMPARISON_RECORDED'")
needle="   assert copycheck['checked'][str(start)]['all_state_exact']"
assert a.count(needle)==1
a=a.replace(needle,needle+"\n   branch=js(prefix+'BRANCH_COPY_EQ.json',{});check=js(prefix+'CHECK_EQ.json',{})\n   assert branch['checked'][str(start)]['all_state_exact']\n   assert check['finite_stage_starts'][str(start)]['exact']\n   if st>6:assert check['device_stage_starts'][str(start+10)]['exact']")
needle="(D/'LOCAL_AUDIT.json').write_text"
pos=a.index(needle)
mutex='''events=[json.loads(x) for x in v['light'].get('ENVIRONMENT_LOCK_EVENTS.jsonl',{}).get('text','').splitlines()]
held=None;seen=set();acquired={'collect':0,'evaluate':0};released={'collect':0,'evaluate':0}
for row in events:
 assert row['kind'] in acquired
 if row['phase']=='acquired':
  assert held is None and row['event_id'] not in seen
  held=row;seen.add(row['event_id']);acquired[row['kind']]+=1
 else:
  assert row['phase']=='released' and held and row['event_id']==held['event_id'] and row['case']==held['case'] and row['kind']==held['kind'] and row['time']>=held['time'] and row['returncode']==0
  released[row['kind']]+=1;held=None
out['environment_mutex_audit']=dict(nonoverlap=True,acquired=acquired,released=released,active=held)
'''
a=a[:pos]+mutex+a[pos:];put('audit_progress.py',a)
put('audit_seed_evaluation.py',read(B/'audit_seed_evaluation.py'))
scripts={}
for case in [g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]:
 s=amounts(read(B/('final_'+case+'_readonly.py')))
 p=json.loads(read(R/case/'PARENT_BINDING.json'))
 lines=s.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith('Q=Path('))
 lines[i]="Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_GOAL41_42_43_20261103_v1/"+case+"');O=Path("+repr(p['output'])+");S=Path("+repr(p['source'])+")";s='\n'.join(lines)+'\n'
 needle="assert len(task6)==1 and len(collections)==40"
 s=s.replace(needle,needle+"\nref=js(Q/'S_REFERENCE.json')['stage6_checkpoint'];assert sha(ref['path'])==ref['sha256']\nreference_model=model(ref['path']);task6_baseline_exact=task6['S']=={k:ah(v.numpy()) for k,v in reference_model.state_dict().items()}\nassert task6_baseline_exact==js(Q/'TASK6_HARDWARE_COMPARISON.json')['all_tensors_exact']")
 s=s.replace('single_S_only=True,','single_S_only=True,task6_baseline_all_tensors_exact=task6_baseline_exact,')
 put('final_'+case+'_readonly.py',s);scripts[case]=s
 agg=read(B/'final_readonly.py');agg=agg[agg.index('"""Only after natural completion:'):]
put('final_readonly.py','CHECK_SCRIPT='+repr(read(R/'check_readonly.py'))+'\nSCRIPTS='+repr(scripts)+'\n'+agg)
a=amounts(read(B/'audit_final_local.py')).replace('OBJECT S41/42/43','GOAL S41/42/43')
a=a.replace('results[key]=dict(',"results[key]=dict(task6_baseline_all_tensors_exact=r['task6_baseline_all_tensors_exact'],")
a=a.replace("assert v['light']['controller.exit']['text']=='0' and not v['processes']","assert v['light']['controller.exit']['text']=='0' and not v['processes']\nassert a['environment_mutex_audit']==dict(nonoverlap=True,acquired={'collect':24,'evaluate':6},released={'collect':24,'evaluate':6},active=None)")
a=a.replace("(D/'LOCAL_FINAL_AUDIT.json').write_text","out['environment_mutex_audit']=a['environment_mutex_audit']\nout['limits'].append('Fresh GOAL3090 runs reuse historical0.1; initial tensor/RNG and paired episodes exact, actual Task6 tensor parity recorded, not hardware-controlled single-factor causality. GOAL engineering validation costs separate from production.')\n(D/'LOCAL_FINAL_AUDIT.json').write_text")
put('audit_final_local.py',a)
c=amounts(read(B/'final_acceptance_20261103_v1/compare_primary_s.py')).replace('parent.parent;D=','parent;D=').replace("binding['sources']['OBJECT']","binding['sources']['GOAL']").replace('OBJECT两个新系数','GOAL两个新系数')
c=c.replace('User-authorized hardware migration and Task6 closed-loop differences despite equal model tensors: no fully hardware-controlled lambda-only causal interpretation','User-authorized hardware migration: record actual Task6 tensor and closed-loop comparison; no fully hardware-controlled lambda-only causal interpretation')
put('compare_primary_s.py',c)
review=dict(status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE',required_actual=dict(checkpoints=24,recoveries=6,teacher_caches=24,initial_noise_episodes=1200,gate_blocks=40200,production_calls=853200),scale_exceptions='NONE; no OBJECT exception inherited',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')})
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
