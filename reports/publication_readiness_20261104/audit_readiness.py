"""Bind existing accepted report sources; no scientific recomputation or remote work."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
W=R.parents[1]
sources=['execution/20261101_ablation_summary/REPORT.md','execution/20261101_ablation_summary/DATA.json','execution/20261101_publication_readiness/REPORT.md','execution/20261101_lambda_r_object_sensitivity/FINAL_REPORT.md','execution/20261103_lambda_r_goal_sensitivity/FINAL_REPORT.md','execution/20261103_lambda_r_goal_sensitivity/GOAL_CLOSURE.json']
bindings={}
for name in sources:
    p=W/name
    raw=p.read_bytes()
    bindings[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
out={'status':'PASS_READONLY_REPORT_BINDINGS','observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_bindings':bindings,'scope':'Review existing accepted reports; no new training, rollout, checkpoint load or acceptance.'}
(R/'EVIDENCE_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(out['status'])
