"""External L005-only acceptance. Does not modify any production or frozen source."""
from pathlib import Path
import ast,json,hashlib,difflib
A=Path(__file__).resolve().parent;R=A.parent;E=R/'engineering_cross_lambda_20261102_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
tree=ast.parse((E/'final_readonly.py').read_text(encoding='utf8'))
values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']}
scripts={}
for case,original in values['SCRIPTS'].items():
 if not case.startswith('L005_'):continue
 s=replace(original,"verify();assert (Q.parent/'engineering_cross_lambda_20261102_v1'/'controller.exit').read_text()=='0'","verify();os.nice(10)\nassert all(js(Q.parent/'engineering_cross_lambda_20261102_v1'/(c+'_evaluate_EXIT.json'))['returncode']==0 for c in ('L005_41','L005_42','L005_43'))")
 s=replace(s,"assert int(p.name)==os.getpid() or str(Q.parent) not in cmd,('Residual',p.name)","assert int(p.name)==os.getpid() or not any(c in cmd for c in ('L005_41','L005_42','L005_43')),('Residual L005',p.name)")
 extra="""sref=js(Q/'S_REFERENCE.json');assert all(sha(x['path'])==x['sha256'] for x in sref.values())
ss=torch.load(sref['stage6_checkpoint']['path'],map_location='cpu')['state_dict'];ss={(k[7:] if k.startswith('module.') else k):ah(v.numpy()) for k,v in ss.items()}
task6_baseline_exact=ss==task6['S']
pb=js(Q/'PARENT_BINDING.json');assert all(sha(p)==h for p,h in pb['files'].items())
"""
 s=replace(s,"records=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()",extra+"records=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()")
 s=replace(s,"single_S_only=True,initial_state_bytes_verified=True","single_S_only=True,task6_baseline_all_tensors_exact=task6_baseline_exact,initial_state_bytes_verified=True")
 compile(s,case,'exec');scripts[case]=s
 (A/(case+'_cpu_source.py')).write_text(s,encoding='utf8')
 (A/(case+'_cpu_source.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='cross/'+case,tofile='L005_only/'+case)),encoding='utf8')
aggregate='''
import subprocess,json,time
def run(script):
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
def scoped(v):
 return {n:x for n,x in v['light'].items() if any(n.startswith(c+'/') for c in SCRIPTS)}
def verify(v):
 assert not v['canceled_mechanism_processes']
 assert not any(any(c in p['cmd'] for c in SCRIPTS) for p in v['processes'])
 boundary={}
 for c in SCRIPTS:
  x=json.loads(v['light']['engineering_cross_lambda_20261102_v1/'+c+'_evaluate_EXIT.json']['text']);assert x['returncode']==0
  assert v['light'][c+'/evaluate.exit']['text'].strip()=='0'
  if c=='L005_43':
   assert v['light'][c+'/train_S.exit']['text'].strip()=='1'
   assert v['light']['engineering_rendezvous_recovery_20261102_v1/L005_43_resume.exit']['text'].strip()=='0'
  else:assert v['light'][c+'/train_S.exit']['text'].strip()=='0'
  assert len(json.loads(v['light'][c+'/OUTCOMES.json']['text']))==200
  boundary[c]=x
 return boundary
v=run(CHECK_SCRIPT);boundary=verify(v);before=scoped(v);results={};diagnostics={}
for case,script in SCRIPTS.items():
 r=run(script)
 assert r['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and r['contract_sha256']==v['children'][case]
 for k,rows in r.pop('diagnostics').items():diagnostics[case+'_'+k]=rows
 for n,x in r.pop('light').items():
  if case+'/'+n in v['light']:assert x['sha256']==v['light'][case+'/'+n]['sha256']
 results[case]=r
after=run(CHECK_SCRIPT);assert verify(after)==boundary
assert after['contract_sha256']==v['contract_sha256'] and scoped(after)==before
v.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',scope='L005_ONLY_L020_CONTINUES',cases=results,diagnostics=diagnostics,group_acceptance_boundary=boundary,group_hashes_before={n:x['sha256'] for n,x in before.items()},group_hashes_after={n:x['sha256'] for n,x in scoped(after).items()},post_audit_observed=after['observed'],post_audit_processes=after['processes'],environment_calls=0,optimizer_updates=0)
print(json.dumps(v))
'''
(A/'final_readonly.py').write_text('CHECK_SCRIPT='+repr(values['CHECK_SCRIPT'])+'\nSCRIPTS='+repr(scripts)+'\n'+aggregate,encoding='utf8')
s=(E/'audit_final_local.py').read_text(encoding='utf8')
s=replace(s,"E=Path(__file__).resolve().parent;R=E.parent;D=Path(sys.argv[1])","R=Path(__file__).resolve().parent.parent;E=R/'engineering_cross_lambda_20261102_v1';D=Path(sys.argv[1])")
start=s.index("a=js(D/'LOCAL_AUDIT.json');");end=s.index('results={}',start)
s=s[:start]+'''a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['scope']=='L005_ONLY_L020_CONTINUES' and v['group_hashes_before']==v['group_hashes_after']
assert not any(any(c in p['cmd'] for c in ('L005_41','L005_42','L005_43')) for p in v['processes']+v['post_audit_processes'])
for case in ('L005_41','L005_42','L005_43'):
 c=a['counts'][case]
 assert (c['epochs'],c['accepted_updates'],c['physical_optimizer_calls'],c['own_isolated_optimizer_calls'],c['own_extra_autograd_calls'],c['training_collection_completed_episodes'],c['formal_episodes'])==(400,105500,195300,91,41,40,200)
 assert v['group_acceptance_boundary'][case]['returncode']==0
assert v['optimizer_updates']==v['environment_calls']==0
'''+s[end:]
s=replace(s,"['L005_41', 'L005_42', 'L005_43', 'L020_41', 'L020_42', 'L020_43']","['L005_41', 'L005_42', 'L005_43']")
s=replace(s,"status='PASS_COMPLETE_LOCAL_ACCEPTANCE'","status='PASS_COMPLETE_L005_LOCAL_ACCEPTANCE',scope='L005_ONLY_L020_PENDING'")
s=replace(s,'epochs=2400,main_updates=633000,physical_training_optimizer_calls=1171800,isolated_optimizer_updates=546,formal_extra_autograd=216,admission_extra_autograd=246,training_collection_episodes=240,formal_episodes=1200,vectors=120,cells=60','epochs=1200,main_updates=316500,physical_training_optimizer_calls=585900,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30')
s=replace(s,"results[key]=dict(gate_audit=", "results[key]=dict(task6_baseline_all_tensors_exact=r['task6_baseline_all_tensors_exact'],gate_audit=")
s=s.replace('Only fixed lambda_R 0.05/0.2 OBJECT S41/42/43; accepted 0.1 read-only. No outcome-based selection.','Only fixed lambda_R 0.05 OBJECT S41/42/43 accepted here; 0.2 continues and 0.1 is read-only. No outcome-based selection.')
(A/'audit_final_local.py').write_text(s,encoding='utf8')
s=(R/'compare_primary_s.py').read_text(encoding='utf8')
for old,new in [('R=Path(__file__).resolve().parent;D=Path(sys.argv[1])','R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])'),("a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==1200","a['status']=='PASS_COMPLETE_L005_LOCAL_ACCEPTANCE' and a['formal_episodes']==600"),("a['main_updates']==633000 and a['isolated_optimizer_updates']==546","a['main_updates']==316500 and a['isolated_optimizer_updates']==273"),("for suite in ['L005','L020']:","for suite in ['L005']:"),("['tensors','rng','hardware','official_cp_sha256','parent_initial_sha256']","['tensors','rng','official_cp_sha256','parent_initial_sha256']"),("assert json.loads(cl[cp+'TASK6_PARITY_S.json']['text'])['same_seed_new_arm_comparator']=='ACCEPTED_S_ALL_TENSORS_EXACT'","assert a['cases'][case]['task6_baseline_all_tensors_exact']\n  hardware=dict(Variant=ci['hardware'],baseline=si['hardware'],initial_hardware_equal=ci['hardware']==si['hardware'],migration='USER_AUTHORIZED_RTX3090')"),('results[case]=dict(Variant_metrics=','results[case]=dict(hardware=hardware,Variant_metrics='),('PASS_NEW_Variant_EXISTING_S_1200_PAIRED_EPISODES_EQUAL_ACCEPTED_UPDATES','PASS_L005_EXISTING_S_600_PAIRED_EPISODES'),('paired_episodes=1200','paired_episodes=600'),('OBJECT两个新系数各三seed，1200回合','OBJECT本次0.05三seed，600回合')]:s=replace(s,old,new)
s=s.replace('Baseline 0.1 historical diagnostics remain separate; the baseline 600 outcomes are reused for two paired comparisons, not 1200 independent baseline outcomes','This partial comparison reuses 600 accepted baseline outcomes once; the future 0.2 comparison does not create independent baseline episodes. Hardware migration and engineering interruptions prevent a fully hardware-controlled lambda-only causal claim.')
(A/'compare_primary_s.py').write_text(s,encoding='utf8')
for p in A.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
(A/'WRAPPER_REVIEW.json').write_text(json.dumps(dict(status='STATIC_ONLY_PENDING_ACTUAL_CPU',scope='L005 only, L020 continues',completion='Three actual evaluate EXIT0, own train exit0 except original L00543 failure plus verified resumed exit0, no target L005 processes',cpu_checkpoints=12,recoveries=3,teacher_caches=12,formal_initial_noise=600,gate_blocks=27300,scientific_mutations=0,optimizer_steps=0,environments=0,source_hashes={str(p.relative_to(R)):sha(p) for p in [E/'final_readonly.py',E/'audit_final_local.py',R/'compare_primary_s.py']},files={p.name:sha(p) for p in A.glob('*.py')}),indent=2)+'\n',encoding='utf8')
print('STATIC_ONLY L005 acceptance wrappers built and compiled; actual CPU pending')
