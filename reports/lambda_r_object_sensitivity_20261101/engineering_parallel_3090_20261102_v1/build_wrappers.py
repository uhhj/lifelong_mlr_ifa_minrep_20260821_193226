"""Local-only external engineering wrappers; no deployment or scientific execution."""
import ast,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent;P=R/'engineering_checks_3090_20261102_v1'
old=P.name;new=E.name
def write(n,s):ast.parse(s);(E/n).write_text(s,encoding='utf8')
write('watchdog.py',(P/'watchdog.py').read_text(encoding='utf8'))
template=(E/'dispatch_template.py').read_text(encoding='utf8');assert template.count('\nSOURCES={}\n')==1
files={n:(E/n).read_text(encoding='utf8') for n in ['controller.py','train_wrapper.py','watchdog.py']}
expected={}
for label,directory,worker,driver in [('probe','lambda_parallel_stage9_20261102_v1','worker.py','driver.py'),('timing','lambda_parallel_timing_20261102_v1','perf_worker.py','perf_driver.py'),('full','lambda_parallel_full_20261102_v1','full_worker.py','full_driver.py')]:
 expected[label]={'/data/mlr_ifa/diagnostics/'+directory+'/'+name:hashlib.sha256((E/local).read_text(encoding='utf8').encode()).hexdigest() for name,local in [('worker.py',worker),('driver.py',driver)]}
template=template.replace('EXPECTED_SOURCES={}','EXPECTED_SOURCES='+repr(expected))
template=template.replace('__EXPECTED_CONTINUATION__',hashlib.sha256((E/'full_continue_v2.py').read_text(encoding='utf8').encode()).hexdigest())
write('dispatch_handoff.py',template.replace('\nSOURCES={}\n','\nSOURCES='+repr(files)+'\n'))
check=(P/'check_readonly.py').read_text(encoding='utf8')
hook="""
P=Q/'engineering_parallel_3090_20261102_v1'
pc=json.loads((P/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in pc['files'].items())
pa=json.loads((P/'PARALLEL_ADMISSION.json').read_text());assert pa['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING' and pa['optimizer_started']==pa['optimizer_completed']==18584 and pa['environments']==pa['extra_autograd']==0
assert not (N/'controller.exit').exists()
engineering.update(prior_checks_contract_sha256=engineering['contract_sha256'],contract_sha256=sha(P/'CONTRACT.json'),frozen_files=len(pc['files']),completion_source='engineering_parallel_3090_20261102_v1/controller.exit',diagnostic_optimizer_calls=34549,parallel_diagnostic_optimizer_calls=18584,parallel_diagnostic_extra_autograd=0,max_training_concurrency=2,collection_serialized=True)
"""
needle="c=json.loads((Q/'CONTRACT.json').read_text());"
assert check.count(needle)==1;check=check.replace(needle,hook+'\n'+needle);write('check_readonly.py',check)
audit=(P/'audit_progress.py').read_text(encoding='utf8').replace(old+'/controller.exit',new+'/controller.exit').replace(old+'/WATCHDOG_RELEASE.json',new+'/WATCHDOG_RELEASE.json').replace('15965','34549')
hook="""
assert 'engineering_checks_3090_20261102_v1/controller.exit' not in v['light']
assert eng['parallel_diagnostic_optimizer_calls']==18584 and eng['parallel_diagnostic_extra_autograd']==0
pa=js('engineering_parallel_3090_20261102_v1/PARALLEL_ADMISSION.json');assert pa['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
assert pa['optimizer_started']==pa['optimizer_completed']==18584 and pa['environments']==0
events=[];acquired={};released={}
for case in ['L005_43','L020_41','L020_42','L020_43']:
 n='engineering_parallel_3090_20261102_v1/'+case+'_COLLECTION_LOCK.jsonl'
 for row in v['light'].get(n,{}).get('text','').splitlines():
  x=json.loads(row);key=(case,int(x['stage']));target=acquired if x['event']=='acquired' else released
  assert x['event'] in ['acquired','released'] and key not in target;target[key]=x
  events.append((x['time'],x['event'],key))
  if x['event']=='released':assert key in acquired and x['returncode']==0
active_collection=None
for when,event,key in sorted(events):
 if event=='acquired':assert active_collection is None;active_collection=key
 else:assert active_collection==key;active_collection=None
for key in released:
 case,st=key;assert js(case+'/MEMORY_S_%d.json'%st)['training_attempts']==10
if formal==1200:assert len(released)==len(acquired)==16 and active_collection is None
root_status=js('STATUS.json',{})
if root_status.get('controller')==json.loads(active.read_text(encoding='utf8'))['controller']:
 running=root_status.get('active',{});assert len(running)<=2
 if len(running)>1:assert all(k.startswith('L020_') and x['phase']=='train' for k,x in running.items())
if any(cc['epochs'] for case,cc in counts.items() if case.startswith('L020_')):
 assert sum(counts['L005_'+str(seed)]['formal_episodes'] for seed in [41,42,43])==600
 assert js('L005_EVALUATION_COMPLETE.json')['episodes']==600
out['parallel_collection_lock_audit']=dict(acquisitions=len(acquired),releases=len(released),active=active_collection,nonoverlap=True)
"""
audit=audit.replace("(D/'LOCAL_AUDIT.json').write_text",hook+"\n(D/'LOCAL_AUDIT.json').write_text");write('audit_progress.py',audit)
record=(P/'record_progress.py').read_text(encoding='utf8').replace(old,new)
record=record.replace('顺序L005三seed训练→600正式→L020三seed训练→600正式。','顺序L005当前42自然完成、43单独训练→600正式→L020三seed最多双任务训练→600串行正式；全部训练采集互斥串行。')
write('record_progress.py',record)
audit_final=(P/'audit_final_local.py').read_text(encoding='utf8').replace(old+'/controller.exit',new+'/controller.exit').replace('15965','34549')
audit_final=audit_final.replace("(D/'LOCAL_FINAL_AUDIT.json').write_text","out['limits'].append('Future L020 training uses two independent processes; collection and evaluation remain serial. Isolated two-seed full-epoch serial/concurrent equality was verified for eight stage/lambda combinations, not every future trajectory.')\n(D/'LOCAL_FINAL_AUDIT.json').write_text")
write('audit_final_local.py',audit_final)
tree=ast.parse((P/'final_readonly.py').read_text(encoding='utf8'));scripts=None
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECK_SCRIPT' for t in node.targets):node.value=ast.Constant(check)
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCRIPTS' for t in node.targets):
  scripts=ast.literal_eval(node.value)
  for case,s in scripts.items():
   needle="Q.parent/'engineering_resume_3090_20261102_v1'/'controller.exit'"
   assert s.count(needle)==1,(case,'Unexpected prior CPU completion source')
   s=s.replace(needle,"Q.parent/'engineering_parallel_3090_20261102_v1'/'controller.exit'")
   ast.parse(s);scripts[case]=s
  node.value=ast.parse(repr(scripts),mode='eval').body
assert scripts and len(scripts)==6
source=ast.unparse(tree).replace("v['light'].get('"+old+"/controller.exit'","v['light'].get('"+new+"/controller.exit'")+'\n'
write('final_readonly.py',source)
for n in ['controller.py','train_wrapper.py','watchdog.py','dispatch_handoff.py','check_readonly.py','audit_progress.py','record_progress.py','final_readonly.py','audit_final_local.py']:ast.parse((E/n).read_text(encoding='utf8'))
review=dict(status='STATIC_ONLY_UNDEPLOYED',completion_source=new+'/controller.exit',production_sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},max_training_concurrency=2,current_training_signals=0,current_run='L005_42 finishes naturally',pool='L020 only',collections='global flock; unchanged command/environment; 16 future ten-episode collections audited for no overlap',evaluation='serial',final_cpu_completion_paths_rebound=6,prior_static_cpu_source_reference_corrected=True,scientific_changes=False)
(E/'STATIC_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps(review))
