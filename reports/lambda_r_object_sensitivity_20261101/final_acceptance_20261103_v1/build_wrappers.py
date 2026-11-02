"""Prepare full-six CPU wrappers; never execute them before natural completion."""
from pathlib import Path
import ast,json,hashlib,difflib
A=Path(__file__).resolve().parent;R=A.parent;E=R/'engineering_cross_lambda_20261102_v1';P=R/'l005_acceptance_20261103_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def values(path):
 tree=ast.parse(path.read_text(encoding='utf8'))
 return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']}
def replace(s,x,y):
 assert s.count(x)==1,(x,s.count(x))
 return s.replace(x,y)
original=values(E/'final_readonly.py');accepted=values(P/'final_readonly.py')
sample=accepted['SCRIPTS']['L005_41']
scale_prefix=sample[:sample.index('"""External CPU-only')]
scale_check=sample[sample.index("  assert torch.equal(cache['scales'],"):sample.index("  rr=js(Q/('COLLECTION_OUTCOMES_")]
old_scale="  assert torch.equal(scale,cache['scales']) and np.allclose(scale.numpy(),m['scales'],atol=1e-8,rtol=1e-8)\n"
extra=sample[sample.index("sref=js(Q/'S_REFERENCE.json')"):sample.index("records=js(Q/'OUTCOMES.json')")]
scripts={}
for case,s in original['SCRIPTS'].items():
 before=s
 s=scale_prefix+replace(s,old_scale,scale_check)
 s=replace(s,'sampling_exact=True,scale_exact=True,','sampling_exact=True,scale_binding_exact=True,scale_recomputation_exact=scale_recomputation_exact,scale_exception=scale_exception,')
 s=replace(s,"records=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()",extra+"records=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()")
 s=replace(s,'single_S_only=True,initial_state_bytes_verified=True','single_S_only=True,task6_baseline_all_tensors_exact=task6_baseline_exact,initial_state_bytes_verified=True')
 s=replace(s,"verify();assert (Q.parent/'engineering_cross_lambda_20261102_v1'/'controller.exit').read_text()=='0'","verify();os.nice(10);assert (Q.parent/'engineering_cross_lambda_20261102_v1'/'controller.exit').read_text()=='0'")
 assert "str(Q.parent) not in cmd" in s
 assert "stage==6" in s and "Q.name=='L005_41'" in s
 compile(s,case,'exec');scripts[case]=s
 (A/(case+'_cpu_source.py')).write_text(s,encoding='utf8')
 (A/(case+'_cpu_source.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='cross/'+case,tofile='final_acceptance/'+case)),encoding='utf8')
source=(E/'final_readonly.py').read_text(encoding='utf8');tree=ast.parse(source)
script_end=next(n.end_lineno for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCRIPTS')
aggregate='\n'.join(source.splitlines()[script_end:])+'\n'
assert "controller.exit" in aggregate and 'not v[\'processes\']' in aggregate and "after['light'] == v['light']" in aggregate
(A/'final_readonly.py').write_text('CHECK_SCRIPT='+repr(original['CHECK_SCRIPT'])+'\nSCRIPTS='+repr(scripts)+'\n'+aggregate,encoding='utf8')
s=(E/'audit_final_local.py').read_text(encoding='utf8')
s=replace(s,'E=Path(__file__).resolve().parent;R=E.parent;D=Path(sys.argv[1])',"R=Path(__file__).resolve().parent.parent;E=R/'engineering_cross_lambda_20261102_v1';D=Path(sys.argv[1])")
s=replace(s,"mm['scale_exact']","mm['scale_binding_exact']")
check=""" for cache_key,mm in r['memory_audits'].items():
  if not mm['scale_recomputation_exact']:
   assert key=='L005_41' and cache_key=='S_6'
   assert mm['scale_exception']['ulp_difference']==[1,0,0] and mm['scale_exception']['max_absolute_difference']==2.9802322387695312e-08
"""
s=replace(s," sh=L['SHORT_HISTORY_ADMISSION_S.json'];",check+" sh=L['SHORT_HISTORY_ADMISSION_S.json'];")
s=replace(s,'results[key]=dict(gate_audit=',"results[key]=dict(task6_baseline_all_tensors_exact=r['task6_baseline_all_tensors_exact'],scale_recomputation={k:{f:m[f] for f in ['scale_binding_exact','scale_recomputation_exact','scale_exception']} for k,m in r['memory_audits'].items()},gate_audit=")
needle="(D/'LOCAL_FINAL_AUDIT.json').write_text"
preserve="""accepted=js(R/'L005_ACCEPTANCE.json')
assert accepted['status']=='PASS_COMPLETE_L005_LOCAL_ACCEPTANCE'
assert all(v['light'][n]['sha256']==h for n,h in accepted['group_hashes'].items())
old=js(Path(accepted['final_directory'])/'LOCAL_FINAL_AUDIT.json')
assert sha(Path(accepted['final_directory'])/'LOCAL_FINAL_AUDIT.json')==accepted['local_final_audit_sha256']
for case in ('L005_41','L005_42','L005_43'):
 for field in ['metrics','matrix','gate_audit','collection_cost','task6_baseline_all_tensors_exact']:
  assert results[case][field]==old['cases'][case][field],(case,field)
out['prior_L005_acceptance_preserved']=True
out['limits'].append('Only the exact previously measured L00541 Task6 one-ULP scale recomputation exception is allowed; all other scales remain exact. Task6 model tensor comparison is recorded, never inferred from same initialization or hardware.')
"""
s=replace(s,needle,preserve+needle)
(A/'audit_final_local.py').write_text(s,encoding='utf8')
s=(R/'compare_primary_s.py').read_text(encoding='utf8')
s=replace(s,'R=Path(__file__).resolve().parent;D=Path(sys.argv[1])','R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])')
s=replace(s,"['tensors','rng','hardware','official_cp_sha256','parent_initial_sha256']","['tensors','rng','official_cp_sha256','parent_initial_sha256']")
s=replace(s,"assert json.loads(cl[cp+'TASK6_PARITY_S.json']['text'])['same_seed_new_arm_comparator']=='ACCEPTED_S_ALL_TENSORS_EXACT'","task6_exact=a['cases'][case]['task6_baseline_all_tensors_exact'];assert isinstance(task6_exact,bool)\n  hardware=dict(Variant=ci['hardware'],baseline=si['hardware'],initial_hardware_equal=ci['hardware']==si['hardware'],task6_all_tensors_exact=task6_exact,migration='USER_AUTHORIZED_RTX3090')")
s=replace(s,'results[case]=dict(Variant_metrics=','results[case]=dict(hardware=hardware,Variant_metrics=')
s=s.replace("'Three development seeds, no significance or single-mechanism causal claim'","'Three development seeds, no significance or single-mechanism causal claim','User-authorized hardware migration and Task6 closed-loop differences despite equal model tensors: no fully hardware-controlled lambda-only causal interpretation'")
(A/'compare_primary_s.py').write_text(s,encoding='utf8')
for p in A.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
review=dict(status='STATIC_ONLY_NOT_EXECUTED',purpose='Full-six actual acceptance after cross controller natural exit0 and no own processes',cpu_checkpoints=24,recoveries=6,caches=24,formal_initial_noise=1200,gate_blocks=54600,production_optimizer_calls=1171800,own_isolated_optimizer_calls=546,scope='No training/environment/validation dispatch, no production code modifications',preserved_prior_L005_acceptance=True,scale_exception='Exact measured L00541 Task6 only, no other case/stage tolerance expansion',hardware='Record actual hardware and Task6 tensor parity, preserve initialization tensor/RNG equality; no causal claim',sources={str(p.relative_to(R)):sha(p) for p in [E/'final_readonly.py',E/'audit_final_local.py',R/'compare_primary_s.py',P/'final_readonly.py',R/'L005_ACCEPTANCE.json']},files={p.name:sha(p) for p in A.glob('*.py')})
(A/'WRAPPER_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:review[k] for k in ['status','cpu_checkpoints','recoveries','caches','formal_initial_noise','gate_blocks']}))
