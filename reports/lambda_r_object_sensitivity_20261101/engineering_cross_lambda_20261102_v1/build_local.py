"""Local composition only. No server calls, no scientific run or deployment."""
import ast,json,hashlib,difflib
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent;prior='engineering_rendezvous_recovery_20261102_v1';P=R/prior;new=E.name
def read(n):return (P/n).read_text(encoding='utf-8-sig')
def save(n,s):
 ast.parse(s);(E/n).write_text(s,encoding='utf8')
 (E/(n+'.diff')).write_text(''.join(difflib.unified_diff(read(n).splitlines(True),s.splitlines(True),fromfile=prior+'/'+n,tofile=new+'/'+n)),encoding='utf8')
extra="""
X=Q/'engineering_cross_lambda_20261102_v1'
xc=json.loads((X/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in xc['files'].items())
xv=json.loads((X/'ENVIRONMENT_LOCK_VALIDATION.json').read_text());assert xv['status']=='PASS_ZERO_ENVIRONMENT_EVALUATION_COLLECTION_MUTEX' and xv['optimizer_calls']==xv['environments']==0
assert not (K/'controller.exit').exists(),'Original recovery coordinator must not claim completion after adoption'
engineering.update(prior_recovery_contract_sha256=engineering['contract_sha256'],contract_sha256=sha(X/'CONTRACT.json'),frozen_files=len(xc['files']),completion_source='engineering_cross_lambda_20261102_v1/controller.exit',cross_lambda_authorized=True,early_L020_case='L020_41',evaluation_collection_mutex=True,cross_lambda_additional_optimizer_calls=0,cross_lambda_additional_environments=0)
"""
s=read('check_readonly.py');needle="c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())";assert s.count(needle)==1;s=s.replace(needle,extra+'\n'+needle);save('check_readonly.py',s)
s=read('audit_progress.py').replace(prior+'/controller.exit',new+'/controller.exit')
old="if len(running)>1:assert all(x['phase']=='train' for x in running.values()) and (set(running)=={'L005_42','L005_43'} or all(k.startswith('L020_') for k in running))"
replacement="""if len(running)>1:
  trains={k for k,x in running.items() if x['phase']=='train'};evals={k for k,x in running.items() if x['phase']=='evaluate'}
  assert len(trains)<=2 and len(evals)<=1
  assert (trains=={'L005_43','L020_41'} and not evals) or (all(k.startswith('L020_') for k in trains) and (not evals or trains<={'L020_41'} and all(k.startswith('L005_') for k in evals)))"""
assert old in s;s=s.replace(old,replacement)
s=s.replace("if any(cc['epochs'] for case,cc in counts.items() if case.startswith('L020_')):","if any(counts[case]['epochs'] for case in ['L020_42','L020_43']):")
needle="(D/'LOCAL_AUDIT.json').write_text"
extra="""
assert eng['cross_lambda_authorized'] and eng['early_L020_case']=='L020_41' and eng['evaluation_collection_mutex']
assert eng['cross_lambda_additional_optimizer_calls']==eng['cross_lambda_additional_environments']==0
assert 'engineering_rendezvous_recovery_20261102_v1/controller.exit' not in v['light']
if 'engineering_cross_lambda_20261102_v1/OWNER_LOST_AFTER_LIVE_COMMIT.json' in v['light']:
 out['status']='ENGINEERING_FAILURE_REQUIRES_REVIEW';out['failures']['cross_lambda_owner_lost']='Keep prior coordinator frozen; audit live L02041 before intervention'
if 'engineering_cross_lambda_20261102_v1/WATCHDOG_RELEASE.json' in v['light']:
 out['status']='ENGINEERING_FAILURE_REQUIRES_REVIEW';out['failures']['cross_lambda_watchdog_released']='No live commitment; prior coordinator released; audit before retry'
cross_launches=[json.loads(x) for x in v['light'].get('engineering_cross_lambda_20261102_v1/LAUNCHES.jsonl',{}).get('text','').splitlines()]
for x in cross_launches:
 if x['phase']=='train':assert '--standalone' not in x['argv'] and '--rdzv_endpoint=localhost:0' in x['argv'] and '--rdzv_id='+x['rdzv_id'] in x['argv']
eval_acquired={};eval_released={};environment_events=list(events)
for case in counts:
 for line in v['light'].get('engineering_cross_lambda_20261102_v1/'+case+'_EVALUATION_LOCK.jsonl',{}).get('text','').splitlines():
  x=json.loads(line);key=('evaluate',case);target=eval_acquired if x['event']=='acquired' else eval_released
  assert x['event'] in ['acquired','released'] and key not in target and x['source']=='parent_lock_unchanged_evaluator';target[key]=x
  environment_events.append((x['time'],x['event'],key))
active_env=None
for when,event,key in sorted(environment_events):
 if event=='acquired':assert active_env is None;active_env=key
 else:assert active_env==key;active_env=None
assert set(eval_released)<=set(eval_acquired)
if formal==1200:assert len(eval_acquired)==len(eval_released)==6 and active_env is None
out['environment_mutex_audit']=dict(nonoverlap=True,collection_acquisitions=len(acquired),collection_releases=len(released),evaluation_acquisitions=len(eval_acquired),evaluation_releases=len(eval_released),active=active_env)
out['rendezvous_audit']['cross_lambda_launches']=len(cross_launches)
out['cross_lambda_authorized']=True
"""
s=s.replace(needle,extra+'\n'+needle);save('audit_progress.py',s)
s=read('audit_final_local.py').replace(prior+'/controller.exit',new+'/controller.exit')
s=s.replace("(D/'LOCAL_FINAL_AUDIT.json').write_text", "out.update(cross_lambda_authorized=True,environment_mutex_audit=a['environment_mutex_audit'])\nout['limits'].append('User-authorized overlap of L00543 and L02041, followed by priority L005 evaluation during L02041 training; all collection/evaluation environments serialized. Scientific per-run budgets and task order unchanged.')\n(D/'LOCAL_FINAL_AUDIT.json').write_text");save('audit_final_local.py',s)
s=read('record_progress.py').replace(prior+'/controller.exit',new+'/controller.exit').replace('本工程'+prior+'/check','本工程'+new+'/check').replace('顺序L00543从第289轮检查点恢复剩余训练→600正式→L020三seed最多双任务训练→600串行正式；全部训练采集互斥串行。','用户授权跨组顺序：不中断L00543并提前并行L02041；L00543自然完成后优先L005600正式，L02041可继续训练，全部正式与采集共用互斥锁；L005正式结束后L02042/43按最多两训练接续，最后L020600串行正式。')
s=s.replace("status=dict(a,status=", "status=dict(a,status=")
save('record_progress.py',s)
tree=ast.parse(read('final_readonly.py'))
for node in tree.body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  if node.targets[0].id=='CHECK_SCRIPT':node.value=ast.Constant((E/'check_readonly.py').read_text(encoding='utf8'));continue
  if node.targets[0].id=='SCRIPTS':
   scripts=ast.literal_eval(node.value)
   for case,s in scripts.items():
    s=s.replace("Q.parent/'"+prior+"'/'controller.exit'","Q.parent/'"+new+"'/'controller.exit'");ast.parse(s);scripts[case]=s
   node.value=ast.parse(repr(scripts),mode='eval').body;continue
 for item in ast.walk(node):
  if isinstance(item,ast.Constant) and item.value==prior+'/controller.exit':item.value=new+'/controller.exit'
save('final_readonly.py',ast.unparse(tree)+'\n')
sources={n:(E/n).read_text(encoding='utf-8-sig') for n in ['controller.py','watchdog.py','evaluate_locked.py']}
for source in sources.values():ast.parse(source)
payload=(E/'dispatch_template.py').read_text(encoding='utf8').replace('SOURCES={}','SOURCES='+repr(sources));ast.parse(payload);(E/'dispatch.py').write_text(payload,encoding='utf8')
review=dict(status='REVIEWED_NOT_DEPLOYED',change='User-authorized cross-lambda schedule and environment serialization; no scientific source mutation',frozen_science_unchanged=True,current_training_signals=0,training_wrapper_reused=prior,independent_launcher_actual_proof=prior+'/LAUNCHER_VALIDATION.json',source_hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},final_cpu_wrappers='STATIC_ONLY',embedded_final_cases=6,special_L00543_recovery_exit_preserved=True,prior_failed_exits_preserved=True,environment_lock_validation='Actual zero-environment diagnostic required by dispatch before any controller signal',max_training=2,early_case='L020_41')
(E/'STATIC_REVIEW.json').write_text(json.dumps(review,indent=2),encoding='utf8');print(json.dumps(review))
