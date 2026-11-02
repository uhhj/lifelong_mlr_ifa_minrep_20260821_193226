"""External live-adoption audit wrappers; no scientific execution."""
import ast,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;P=E.parent/'engineering_parallel_3090_20261102_v1';old=P.name;new=E.name
def write(n,s):ast.parse(s);(E/n).write_text(s,encoding='utf8')
sources={n:(E/n).read_text(encoding='utf8') for n in ['controller.py','watchdog.py','train_wrapper.py','collection_startup_hook.py']}
template=(E/'dispatch_template.py').read_text(encoding='utf8');assert template.count('\nSOURCES={}\n')==1
write('dispatch_live.py',template.replace('\nSOURCES={}\n','\nSOURCES='+repr(sources)+'\n'))
s=(P/'check_readonly.py').read_text(encoding='utf8');needle="c=json.loads((Q/'CONTRACT.json').read_text());";assert s.count(needle)==1
hook="""
L=Q/'engineering_live_parallel_3090_20261102_v1'
lc=json.loads((L/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in lc['files'].items())
lv=json.loads((L/'STARTUP_LOCK_VALIDATION.json').read_text());assert lv['status']=='PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK' and lv['optimizer_calls']==lv['environments']==0
assert not (P/'controller.exit').exists()
engineering.update(prior_parallel_contract_sha256=engineering['contract_sha256'],contract_sha256=sha(L/'CONTRACT.json'),frozen_files=len(lc['files']),completion_source='engineering_live_parallel_3090_20261102_v1/controller.exit',live_adoption=True,startup_collection_lock=True,additional_optimizer_calls=0,additional_environments=0)
"""
s=s.replace(needle,hook+'\n'+needle);write('check_readonly.py',s)
audit=(P/'audit_progress.py').read_text(encoding='utf8').replace(old+'/controller.exit',new+'/controller.exit').replace(old+'/WATCHDOG_RELEASE.json',new+'/WATCHDOG_RELEASE.json')
audit=audit.replace("for case in ['L005_43','L020_41','L020_42','L020_43']:","for case in ['L005_42','L005_43','L020_41','L020_42','L020_43']:")
audit=audit.replace("n='"+old+"/'+case+'_COLLECTION_LOCK.jsonl'","n='"+new+"/'+case+'_COLLECTION_LOCK.jsonl'")
audit=audit.replace("if x['event']=='released':assert key in acquired and x['returncode']==0","if x['event']=='released':assert key in acquired and x['source']=='pre_import_startup_hook'")
audit=audit.replace("if formal==1200:assert len(released)==len(acquired)==16 and active_collection is None", "expected={(case,int(st)) for case,stages in js('"+new+"/GUARD.json')['expected_collection_stages'].items() for st in stages}\nassert set(acquired)<=expected\nif formal==1200:assert set(released)==set(acquired)==expected and active_collection is None")
audit=audit.replace("if len(running)>1:assert all(k.startswith('L020_') and x['phase']=='train' for k,x in running.items())","if len(running)>1:assert all(x['phase']=='train' for x in running.values()) and (set(running)=={'L005_42','L005_43'} or all(k.startswith('L020_') for k in running))")
needle="(D/'LOCAL_AUDIT.json').write_text"
extra="""
assert 'engineering_parallel_3090_20261102_v1/controller.exit' not in v['light']
if 'engineering_live_parallel_3090_20261102_v1/OWNER_LOST_AFTER_LIVE_COMMIT.json' in v['light']:
 out['status']='ENGINEERING_FAILURE_REQUIRES_REVIEW';out['failures']['live_owner_lost']='Keep frozen waiters; never duplicate current runs'
if 'engineering_live_parallel_3090_20261102_v1/LIVE_COMMITTED.json' in v['light']:
 assert js('engineering_live_parallel_3090_20261102_v1/LIVE_COMMITTED.json')['training_signals']==0
out['parallel_collection_lock_audit']['expected_total']=len(expected)
"""
audit=audit.replace(needle,extra+'\n'+needle);write('audit_progress.py',audit)
record=(P/'record_progress.py').read_text(encoding='utf8').replace(old,new)
record=record.replace('L005当前42自然完成、43单独训练','L005当前42原进程与43并行训练')
write('record_progress.py',record)
final=(P/'audit_final_local.py').read_text(encoding='utf8').replace(old+'/controller.exit',new+'/controller.exit').replace('Future L020 training uses two independent processes','L00542/43 and future L020 training use at most two independent processes')
write('audit_final_local.py',final)
tree=ast.parse((P/'final_readonly.py').read_text(encoding='utf8'))
for n in tree.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
  if n.targets[0].id=='CHECK_SCRIPT':n.value=ast.Constant(s)
  if n.targets[0].id=='SCRIPTS':
   scripts=ast.literal_eval(n.value)
   for case,script in scripts.items():
    oldpath="Q.parent/'"+old+"'/'controller.exit'";assert script.count(oldpath)==1
    scripts[case]=script.replace(oldpath,"Q.parent/'"+new+"'/'controller.exit'");ast.parse(scripts[case])
   n.value=ast.parse(repr(scripts),mode='eval').body
write('final_readonly.py',ast.unparse(tree).replace("v['light'].get('"+old+"/controller.exit'","v['light'].get('"+new+"/controller.exit'")+'\n')
review=dict(status='STATIC_ONLY_PENDING_ACTUAL_DEPLOYMENT',sources={n:hashlib.sha256(x.encode()).hexdigest() for n,x in sources.items()},current_training_signals=0,immediate_case='L005_43',existing_process='L005_42 unchanged',parallel_training_admission='Reuse already actual8-stage full-epoch serial/concurrent validation, no repeated optimizer diagnostics',startup_validation='Zero optimizer/environment; exact path and random values; lock serialized; disabled/unmatched startup unchanged',final_CPU_completion_source=new+'/controller.exit',old_completion_sources_not_fabricated=True)
(E/'STATIC_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8');print(json.dumps(review))
