"""Record actual startup only, no scientific or remote actions."""
from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261109_154009'
js=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=js(D/'LOCAL_AUDIT.json');e=js(R/'ACTIVE_ROOT.json');bundle=js(R/'DEPLOY_TEXT_BUNDLE.json')
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  n=f'{suite}/Q1_{seed}/PROTOCOL.md';(R/n).write_text(bundle[n],encoding='utf8',newline='\n')
old=R/'PREPARATION_STATUS.json';history=R/'INITIAL_CAPACITY_BLOCK_20261109.json'
assert not history.exists();shutil.copyfile(old,history)
status=dict(status='PARTIAL_RUNNING',observation=a['observation'],active_suite='OBJECT',coordinator=e['controller'],coordinator_contract=e['contract'],suites=e['suites'],counts=a['counts'],new_total_epochs=a['epochs'],new_total_physical_optimizer_calls=a['physical_training_optimizer_calls'],new_total_isolated_optimizer_calls=a['own_isolated_optimizer_calls'],new_total_admission_autograd=a['own_extra_autograd_calls'],new_total_collection=0,new_total_formal=0,goal_status='DEPLOYED_QUEUED_NOT_STARTED',failures=a['failures'],evidence=str(D),local_payload_archive=False,old_S_retrained=False,old_S_reevaluated=False,final_acceptance='PENDING',science_budget=dict(epochs=2400,accepted_updates=544800,physical_optimizer_calls=1012500,gate_blocks=47400,isolated_optimizer_calls=546,admission_autograd=246,direction_autograd_max=216,collection=240,formal=1200))
for name in ['PREPARATION_STATUS.json','RUN_STATUS.json']:(R/name).write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=status['status'],coordinator=e['controller'],epochs=a['epochs'])))
