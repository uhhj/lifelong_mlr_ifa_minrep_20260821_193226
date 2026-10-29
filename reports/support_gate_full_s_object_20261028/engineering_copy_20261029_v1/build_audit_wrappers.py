"""External acceptance adapters bind the actual engineering completion source."""
from pathlib import Path
import ast,json,hashlib
E=Path(__file__).resolve().parent;N=E.parent
completion='engineering_copy_20261029_v1/controller.exit'
check=(N/'check_readonly.py').read_text(encoding='utf8')
needle='procs=[];oldprocs=[]'
extra="""E=Q/'engineering_copy_20261029_v1'
ec=json.loads((E/'CONTRACT.json').read_text())
assert ec['scientific_contract_sha256']==sha(Q/'CONTRACT.json')
assert all(sha(p)==h for p,h in ec['files'].items())
assert sha(E/'VALIDATION.json')==ec['validation_sha256']
assert not (Q/'controller.exit').exists(),'Original coordinator did not finish this queue'
"""
assert needle in check;check=check.replace(needle,extra+needle)
check=check.replace("source_files=len(c['files']),", "source_files=len(c['files']),engineering_contract_sha256=sha(E/'CONTRACT.json'),engineering_files=len(ec['files']),completion_source='engineering_copy_20261029_v1/controller.exit',")
(E/'check_readonly.py').write_text(check,encoding='utf8')
audit=(N/'audit_progress.py').read_text(encoding='utf8')
audit=audit.replace('R=Path(__file__).resolve().parent;','R=Path(__file__).resolve().parent.parent;')
audit=audit.replace("v['light'].get('controller.exit',{})", "v['light'].get('"+completion+"',{})")
needle='out=dict(status='
extra="""eng=js('engineering_copy_20261029_v1/VALIDATION.json')
assert eng['status']=='PASS_FULL_TRAINING_PATH_EXACT' and eng['optimizer_calls_started']==eng['optimizer_calls_completed']==3914
assert eng['environments']==eng['accepted_scientific_epochs']==0 and eng['completed_seed41_artifacts_unchanged']
assert len(eng['stages'])==4 and [x['batches'] for x in eng['stages']]==[157,227,309,362]
assert v['engineering_files']==5 and v['completion_source']=='engineering_copy_20261029_v1/controller.exit'
engineering=dict(contract=v['engineering_contract_sha256'],diagnostic_optimizer_calls=3914,diagnostic_extra_autograd_calls=eng['extra_autograd_calls'],diagnostic_environments=0,completion_source=v['completion_source'],validation_scope=eng['scope'])
"""
assert needle in audit;audit=audit.replace(needle,extra+needle).replace("counts=counts,epochs=epochs", "engineering=engineering,counts=counts,epochs=epochs")
(E/'audit_progress.py').write_text(audit,encoding='utf8')
source=(N/'final_readonly.py').read_text(encoding='utf8');tree=ast.parse(source);scripts={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='SCRIPTS' for x in node.targets):scripts=ast.literal_eval(node.value)
assert set(scripts)=={'41','42','43'}
for seed,s in scripts.items():
 s=s.replace("assert (Q.parent/'controller.exit').read_text()=='0'", "assert (Q.parent/'"+completion+"').read_text()=='0' and not (Q.parent/'controller.exit').exists() and (Q.parent/'engineering_copy_20261029_v1/HANDOFF_COMPLETE.json').exists()")
 scripts[seed]=s;(E/('final_seed'+seed+'_readonly.py')).write_text(s,encoding='utf8')
# Replace only the embedded read-only scripts and real completion marker.
lines=source.splitlines(True)
replacements=[]
for node in tree.body:
 if isinstance(node,ast.Assign):
  names=[x.id for x in node.targets if isinstance(x,ast.Name)]
  if 'CHECK_SCRIPT' in names:replacements.append((node.lineno-1,node.end_lineno,'CHECK_SCRIPT='+repr(check)+'\n'))
  if 'SCRIPTS' in names:replacements.append((node.lineno-1,node.end_lineno,'SCRIPTS='+repr(scripts)+'\n'))
for start,end,new in sorted(replacements,reverse=True):lines[start:end]=[new]
source=''.join(lines).replace("v['light'].get('controller.exit',{})", "v['light'].get('"+completion+"',{})")
(E/'final_readonly.py').write_text(source,encoding='utf8')
final=(N/'audit_final_local.py').read_text(encoding='utf8')
final=final.replace('R=Path(__file__).resolve().parent;', 'R=Path(__file__).resolve().parent.parent;')
final=final.replace("str(R/'audit_progress.py')", "str(R/'engineering_copy_20261029_v1/audit_progress.py')")
final=final.replace("v['light']['controller.exit']", "v['light']['"+completion+"']")
final=final.replace("results={}\n", "assert 'controller.exit' not in v['light']\nassert json.loads(v['light']['engineering_copy_20261029_v1/HANDOFF_COMPLETE.json']['text'])['root_lock_acquired']\nassert set(json.loads(v['light']['engineering_copy_20261029_v1/SEED43_CAPTURE_EQ.json']['text'])['checked'])=={'0','15700','38400','69300'}\nresults={}\n",1)
final=final.replace("out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',", "out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',engineering=a['engineering'],")
(E/'audit_final_local.py').write_text(final,encoding='utf8')
names=['check_readonly.py','audit_progress.py','final_readonly.py','audit_final_local.py']+['final_seed'+s+'_readonly.py' for s in scripts]
for name in names:ast.parse((E/name).read_text(encoding='utf8'),feature_version=(3,8))
review=dict(status='STATIC_ONLY_FINAL_CPU_NOT_EXECUTED',completion_source=completion,files={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in names},scientific_budget_unchanged=True,diagnostic_cost_separate=True,final_CPU_runs=0)
(E/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps(review))
