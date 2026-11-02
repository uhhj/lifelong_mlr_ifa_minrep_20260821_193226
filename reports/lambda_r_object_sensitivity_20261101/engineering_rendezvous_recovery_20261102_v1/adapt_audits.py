"""External audit adapters: preserve exact original failures and identify true recovery completion."""
from pathlib import Path
import ast,json,hashlib,difflib
E=Path(__file__).resolve().parent;R=E.parent;old='engineering_live_parallel_3090_20261102_v1';new=E.name;O=R/old
def read(n):return (O/n).read_text(encoding='utf-8-sig')
def save(n,s):
 ast.parse(s);(E/n).write_text(s,encoding='utf8')
 (E/(n+'.diff')).write_text(''.join(difflib.unified_diff(read(n).splitlines(True),s.splitlines(True),fromfile=old+'/'+n,tofile=new+'/'+n)),encoding='utf8')
extra="""
K=Q/'engineering_rendezvous_recovery_20261102_v1'
kc=json.loads((K/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in kc['files'].items())
kv=json.loads((K/'VALIDATION.json').read_text());assert kv['status']=='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK' and kv['started']==kv['completed']==869 and kv['extra_autograd']==11 and kv['environments']==0
assert len(kv['comparisons'])==21 and all(x['original_gate_exact'] for x in kv['comparisons'][:20]) and kv['comparisons'][-1]['next_unsaved_block']
assert kv['full_state_restore_exact'] and kv['save_load_roundtrip_exact'] and kv['production_metadata_unchanged'] and kv['restored_after_diagnostics']
kl=json.loads((K/'LAUNCHER_VALIDATION.json').read_text());assert kl['status']=='PASS_INDEPENDENT_EPHEMERAL_RENDEZVOUS_LIFECYCLE' and kl['early_peer_exit_survived'] and kl['optimizer_calls']==kl['environments']==0
mitigated=json.loads((K/'MITIGATED_FAILURES.json').read_text());assert set(mitigated)=={'L005_43/train_S.exit','engineering_live_parallel_3090_20261102_v1/controller.exit','engineering_live_parallel_3090_20261102_v1/FAILURE.json'}
assert all(sha(Q/n)==h for n,h in mitigated.items())
interruption=json.loads((K/'INTERRUPTION.json').read_text());assert all(sha(p)==h for p,h in interruption['completed_hashes'].items())
assert sha(interruption['checkpoint_backup'])==interruption['checkpoint_sha256']
assert json.loads((Q/'L005_43/EPOCHS_S.json').read_text())[:289]==json.loads((K/'ORIGINAL_EPOCHS.json').read_text())
assert (Q/'L005_43/GATES.jsonl').read_bytes().startswith((K/'INTERRUPTED_GATES.jsonl').read_bytes())
engineering.update(prior_live_contract_sha256=engineering['contract_sha256'],contract_sha256=sha(K/'CONTRACT.json'),frozen_files=len(kc['files']),completion_source='engineering_rendezvous_recovery_20261102_v1/controller.exit',diagnostic_optimizer_calls=35418,diagnostic_extra_autograd=94,rendezvous_recovery_optimizer_calls=869,rendezvous_recovery_extra_autograd=11,rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,rendezvous_partial_unrecorded_cost='UNKNOWN',mitigated_failure_hashes=mitigated,independent_rendezvous_verified=True)
"""
s=read('check_readonly.py');needle="c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())"
assert s.count(needle)==1;s=s.replace(needle,extra+'\n'+needle).replace('12*1024**3+346429749','12*1024**3+346429749+346445621');save('check_readonly.py',s)
s=read('audit_progress.py').replace(old+'/controller.exit',new+'/controller.exit').replace('==34549','==35418').replace('==83','==94').replace('optimizer_calls=34549','optimizer_calls=35418')
needle="for n,x in v['light'].items():\n if n.endswith('.exit')"
replace="""mitigated=js('engineering_rendezvous_recovery_20261102_v1/MITIGATED_FAILURES.json')
assert set(mitigated)=={'L005_43/train_S.exit','engineering_live_parallel_3090_20261102_v1/controller.exit','engineering_live_parallel_3090_20261102_v1/FAILURE.json'}
for n,h in mitigated.items():assert v['light'][n]['sha256']==h
for n,x in v['light'].items():
 if n in mitigated:continue
 if n.endswith('.exit')"""
assert s.count(needle)==1;s=s.replace(needle,replace)
for prior,suffix in [('engineering_resume_3090_20261102_v1','TASK6_GPU_COMPARISON'),('engineering_resume_3090_20261102_v1','BRANCH_COPY_EQ'),('engineering_checks_3090_20261102_v1','CHECK_EQ')]:
 needle="js('"+prior+"/'+case+'_"+suffix+".json')";assert needle in s
 s=s.replace(needle,"js('"+new+"/'+case+'_"+suffix+".json',"+needle+")")
needle="(D/'LOCAL_AUDIT.json').write_text"
extra="""
assert eng['rendezvous_recovery_optimizer_calls']==869 and eng['rendezvous_recovery_extra_autograd']==11 and eng['rendezvous_partial_recorded_optimizer_calls']==400 and eng['total_recorded_abandoned_partial_optimizer_calls']==540
assert eng['rendezvous_partial_unrecorded_cost']=='UNKNOWN' and eng['independent_rendezvous_verified']
out.update(mitigated_failures_preserved=mitigated,rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,rendezvous_partial_unrecorded_cost='UNKNOWN')
launches=[json.loads(x) for x in v['light'].get('engineering_rendezvous_recovery_20261102_v1/LAUNCHES.jsonl',{}).get('text','').splitlines()]
for x in launches:
 if x['phase']=='train':
  assert '--standalone' not in x['argv'] and '--rdzv_endpoint=localhost:0' in x['argv'] and '--rdzv_backend=c10d' in x['argv'] and '--rdzv_id='+x['rdzv_id'] in x['argv']
running=js('STATUS.json',{}).get('active',{})
actual=[js('engineering_rendezvous_recovery_20261102_v1/'+case+'_RENDEZVOUS.json') for case,x in running.items() if x['phase']=='train']
actual=[x for x in actual if x is not None]
assert all(x['world_size']==1 for x in actual)
assert len({x['master_port'] for x in actual})==len(actual)
out['rendezvous_audit']=dict(launches=len(launches),observed_training_endpoints=actual,independent_endpoint_arguments=True)
"""
s=s.replace(needle,extra+'\n'+needle);save('audit_progress.py',s)
s=read('audit_final_local.py').replace(old+'/controller.exit',new+'/controller.exit').replace('optimizer_calls=34549','optimizer_calls=35418')
needle="assert L['train_S.exit']==L['evaluate.exit']=='0' and L['TRAIN_S.json']['gradient_autograd_calls']==36"
assert needle in s;s=s.replace(needle,"assert L['evaluate.exit']=='0' and L['TRAIN_S.json']['gradient_autograd_calls']==36\n if key=='L005_43':assert L['train_S.exit']=='1' and v['light']['"+new+"/L005_43_resume.exit']['text']=='0'\n else:assert L['train_S.exit']=='0'")
s=s.replace("out['limits'].append('L00542/43", "out.update(rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,rendezvous_partial_unrecorded_cost='UNKNOWN',mitigated_failures_preserved=a['mitigated_failures_preserved'])\nout['limits'].append('Rendezvous interruption at L00543 epoch289: 400 recorded abandoned candidate calls plus UNKNOWN unlogged cost; 869 isolated optimizer/11 autograd exact recovery diagnostics are separate. Original failed exits remain preserved.')\nout['limits'].append('L00542/43")
save('audit_final_local.py',s)
s=read('record_progress.py').replace(old,new).replace('顺序L005当前42原进程与43并行训练','顺序L00543从第289轮检查点恢复剩余训练').replace("status.update(hardware_migration=", "status.update(rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,rendezvous_partial_unrecorded_cost='UNKNOWN',mitigated_failures_preserved=a['mitigated_failures_preserved'])\nstatus.update(hardware_migration=")
save('record_progress.py',s)
tree=ast.parse(read('final_readonly.py'));values={}
for n in tree.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']:values[n.targets[0].id]=ast.literal_eval(n.value)
scripts=values['SCRIPTS'];needle="assert all((Q/('train_'+a+'.exit')).read_text()=='0' for a in ARMS) and (Q/'evaluate.exit').read_text()=='0'"
for case,s in scripts.items():
 s=s.replace("Q.parent/'"+old+"'/'controller.exit'","Q.parent/'"+new+"'/'controller.exit'")
 if case=='L005_43':
  assert s.count(needle)==1;s=s.replace(needle,"assert ARMS==('S',) or list(ARMS)==['S']\nassert (Q/'train_S.exit').read_text()=='1' and (Q.parent/'"+new+"'/'L005_43_resume.exit').read_text()=='0' and (Q/'evaluate.exit').read_text()=='0'")
 ast.parse(s);scripts[case]=s
for n in tree.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
  if n.targets[0].id=='CHECK_SCRIPT':n.value=ast.Constant((E/'check_readonly.py').read_text(encoding='utf8'))
  if n.targets[0].id=='SCRIPTS':n.value=ast.parse(repr(scripts),mode='eval').body
for node in tree.body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']:continue
 for item in ast.walk(node):
  if isinstance(item,ast.Constant) and item.value==old+'/controller.exit':item.value=new+'/controller.exit'
s=ast.unparse(tree)+'\n';save('final_readonly.py',s)
review=dict(status='STATIC_ONLY_NOT_FINAL_ACCEPTANCE',source='Actual launcher lifecycle and GPU recovery proof independently passed; final CPU awaits all six runs',embedded_cpu_cases=len(scripts),preserved_failed_exit=True,completion_source=new+'/controller.exit',new_diagnostic_optimizer=869,new_diagnostic_autograd=11,total_diagnostic_optimizer=35418,total_diagnostic_autograd=94,total_recorded_abandoned_partial_optimizer=540,unrecorded_cost='UNKNOWN',files={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['check_readonly.py','audit_progress.py','record_progress.py','final_readonly.py','audit_final_local.py']})
(E/'AUDIT_WRAPPER_REVIEW.json').write_text(json.dumps(review,indent=2),encoding='utf8')
print(json.dumps(review))
