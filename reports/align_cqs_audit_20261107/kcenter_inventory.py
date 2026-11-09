"""Recheck historical K-Center AUC from existing matrices; no scientific runs."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
R=Path(__file__).resolve().parent;W=R.parents[1]
op=W/'execution/20261017_five_seed_summary/RESULTS.json';gp=W/'execution/20261017_coreset_goal_three_seed/FINAL_RECOMPUTED.json'
od=json.loads(op.read_text(encoding='utf8'));gd=json.loads(gp.read_text(encoding='utf8'))
def auc(m):return sum(sum(Fraction(str(m[s][t])) for s in range(t,4))/(4-t) for t in range(4))/4
rows=[]
for r in od['first_runs']:
 if r['seed'] in [42,43,44] and r['method'] in ['random','coreset']:
  a=auc(r['matrix']);assert a==100*Fraction(r['metrics']['AUC']['fraction']);rows.append(dict(suite='OBJECT',seed=r['seed'],method=r['method'],AUC=float(a)))
for s,case in gd['seeds'].items():
 for method in ['random','coreset']:
  r=case[method];a=auc(r['matrix']);assert a==100*Fraction(r['metrics']['AUC']['fraction']);rows.append(dict(suite='GOAL',seed=int(s),method=method,AUC=float(a)))
assert len(rows)==12
summary={suite:{method:sum(r['AUC'] for r in rows if r['suite']==suite and r['method']==method)/3 for method in ['random','coreset']} for suite in ['OBJECT','GOAL']}
paths=[op,gp,R/'KCENTER_INVENTORY_20261109.md',W/'execution/20261107_ra_selection_visualization/DISTANCE_REPORT.md',W/'execution/20261013_coreset_extension/FINAL_REPORT.md',W/'execution/20261017_coreset_goal_three_seed/FINAL_REPORT.md']
out=dict(status='PASS_READONLY_KCENTER_INVENTORY',sources={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},rows=rows,summary=summary,training_calls=0,forward_calls=0,environment_calls=0)
(R/'KCENTER_INVENTORY_20261109.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(summary))
