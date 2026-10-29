"""Compose external final auditors. Static review only, no scientific operations."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent;N=R.parent/'20261028_support_gate_full_s_object'
OLD='SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1';NEW='SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1'
def adapt(s):
 for a,b in [(OLD,NEW),('105500','76100'),('195300','142200'),('89800','66100'),('316500','228300'),('585900','426600'),('269400','198300'),('9100','6700'),('27300','20100')]:s=s.replace(a,b)
 return s
def put(n,s):
 ast.parse(s,feature_version=(3,8));(R/n).write_text(s,encoding='utf8')
scripts={}
for seed in (41,42,43):
 p=json.loads((R/f'seed{seed}/PARENT_BINDING.json').read_text(encoding='utf8'))
 s=adapt((N/f'final_seed{seed}_readonly.py').read_text(encoding='utf8'))
 line=next(x for x in s.splitlines() if x.startswith('Q=Path('))
 s=s.replace(line,"Q=Path('/data/mlr_ifa/experiment_workspaces/"+NEW+f"/seed{seed}');O=Path("+repr(p['output'])+");S=Path("+repr(p['source'])+")")
 s=s.replace("verify();assert (Q.parent/'controller.exit').read_text()=='0'", "verify();assert (Q.parent/'controller.exit').read_text()=='0'\ncopyeq=js(Q/'COPY_EQ.json');assert set(copyeq['checked'])=={'0','10000','23900','42400'} and all(x['all_state_exact'] and x['optimizer_calls_added']==x['environments_added']==0 for x in copyeq['checked'].values())")
 put(f'final_seed{seed}_readonly.py',s);scripts[str(seed)]=s
s=(N/'final_readonly.py').read_text(encoding='utf8')
start=s.index('CHECK_SCRIPT=');end=s.index('\ndef run(',start)
s=s[:start]+'CHECK_SCRIPT='+repr((R/'check_readonly.py').read_text(encoding='utf8'))+'\nSCRIPTS='+repr(scripts)+'\n'+s[end:]
put('final_readonly.py',s)
for n in ['audit_final_local.py','audit_seed_evaluation.py','describe_final.py']:
 s=adapt((N/n).read_text(encoding='utf8')).replace('S OBJECT','S GOAL')
 put(n,s)
files=['final_readonly.py','audit_final_local.py','audit_seed_evaluation.py','describe_final.py']+[f'final_seed{s}_readonly.py' for s in (41,42,43)]
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(dict(status='STATIC_ONLY_NOT_EXECUTED_FINAL_ACCEPTANCE_PENDING',generated={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},actual_final_CPU_checks=0,training_updates=0,environment_calls=0,requires='Natural controller.exit0 and no own processes, then real CPU audits and local acceptance'),indent=2)+'\n',encoding='utf8')
print('STATIC_FINAL_WRAPPERS_READY_NOT_EXECUTED')
