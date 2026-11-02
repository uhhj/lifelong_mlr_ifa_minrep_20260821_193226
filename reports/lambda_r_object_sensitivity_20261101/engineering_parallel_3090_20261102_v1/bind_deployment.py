"""Bind one actual deployment receipt locally, before independent progress audit."""
import json,sys,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'))
r=js(D/'EVIDENCE.json');assert r['status']=='ENGINEERING_DISPATCHED_NOT_YET_HANDOFF'
assert hashlib.sha256(json.dumps(r['contract'],indent=2).encode()).hexdigest()==r['contract_sha256']
assert r['contract']['version']==E.name and r['contract']['diagnostic_optimizer_calls']==34549 and r['contract']['diagnostic_extra_autograd']==83
review=js(E/'STATIC_REVIEW.json')
for n,h in review['production_sources'].items():assert r['contract']['files'][str(Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')/E.name/n).replace('\\','/')]==h
assert js(E/'PARALLEL_TRIAL_AUDIT.json')['status']=='PASS_LOCAL_PARALLEL_DIAGNOSTICS'
for name in ['ACTIVE_ROOT.json','MONITOR_HANDOFF.md','REPORT.md','FINAL_WRAPPER_REVIEW.json']:
 p=E/('PRE_HANDOFF_'+name);assert not p.exists();p.write_bytes((R/name).read_bytes())
a=js(R/'ACTIVE_ROOT.json');assert a['controller']==4347 and a['engineering_directory']=='engineering_checks_3090_20261102_v1'
a.update(prior_checks_controller=a['controller'],prior_checks_contract=a['engineering_contract'],controller=r['pid'],engineering_contract=r['contract_sha256'],engineering_directory=E.name,engineering_dispatch_evidence=D.name,completion_source=E.name+'/controller.exit',engineering_frozen_files=len(r['contract']['files']),engineering_state='DISPATCHED_PENDING_ACTUAL_ARMED_AUDIT',watchdog=None,max_training_concurrency=2,parallel_group='L020')
(R/'ACTIVE_ROOT.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf8')
(E/'DISPATCH_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
f=js(R/'FINAL_WRAPPER_REVIEW.json');f.update(engineering_completion_source=E.name+'/controller.exit',engineering_contract=r['contract_sha256'],total_engineering_diagnostic_optimizer=34549,total_engineering_diagnostic_autograd=83,parallel_diagnostic_optimizer_calls=18584,parallel_diagnostic_autograd=0,engineering_final_status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE',all_six_embedded_CPU_completion_sources_adapted=True,parallel_collection_lock_audit=True)
f['engineering_wrappers']={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['check_readonly.py','audit_progress.py','record_progress.py','final_readonly.py','audit_final_local.py']}
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(f,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='ACTUAL_DEPLOYMENT_BOUND_PENDING_PROCESS_AUDIT',controller=r['pid'],contract=r['contract_sha256'],frozen_files=len(r['contract']['files']))))
