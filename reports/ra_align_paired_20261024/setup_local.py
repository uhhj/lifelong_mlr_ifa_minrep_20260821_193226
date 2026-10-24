from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
P=R.parent/'20261024_align_behavior_drift'
A=R.parent/'20261022_align_object_extension_goal_conditional'
spec=json.loads((P/'INPUT_SPEC.json').read_text())
for e in spec['seeds']:
 pkg=next(A.glob('final_GOAL_'+str(e['seed'])+'_*'))
 en=json.loads((pkg/'launcher/ENTRY.json').read_text())
 e['ra_output']=en['parent'];e['ra_contract_sha256']=en['parent_contract_sha256']
 e['local_package']=str(pkg)
(R/'INPUT_SPEC.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
for n in ['run_readonly.py','publish.py']:
 t=(P/n).read_text(encoding='utf8').replace('reports/align_behavior_drift_20261024','reports/ra_align_paired_20261024').replace('Audit ALIGN GOAL retained-window behavioral drift','Audit RA and ALIGN common-window and paired trajectory diagnostics')
 (R/n).write_text(t,encoding='utf8')
