"""Read accepted scalar files only; paired demo comparison and honest train/test aggregates."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;E=R/'final_scalars_readonly_20261023_160059';S=R/'snapshot_20261023_160028';F=R/'final_A_SPATIAL_20261023'
audit=json.loads((F/'LOCAL_FINAL_AUDIT.json').read_text());assert audit['status']=='PASS_LOCAL_COMPLETE_A'
def js(p):return json.loads(p.read_text())
rows=[]
for task in (6,7):
 g=js(E/('G_task%d.json'%task));s=js(E/('S_task%d.json'%task));assert g['ids']==s['ids']
 ge=np.array(g['modes']['normal']['error_mm']);se=np.array(s['modes']['normal']['error_mm']);d=np.array(g['demo_ids'])
 for j,target in enumerate(('object_eef','object_goal')):
  delta=np.array([np.median(ge[d==demo,j])-np.median(se[d==demo,j]) for demo in sorted(set(d))]);rng=np.random.RandomState(410023);ci=np.quantile(delta[rng.randint(0,10,(10000,10))].mean(1),[.025,.975])
  rows.append(dict(task=task,target=target,positive_means_patch_better=True,paired_mean_demo_median_improvement_mm=float(delta.mean()),exploratory_cluster_ci95_mm=ci.tolist(),global_median_mm=float(np.median(ge[:,j])),patch_median_mm=float(np.median(se[:,j])),global_p90_mm=float(np.quantile(ge[:,j],.9)),patch_p90_mm=float(np.quantile(se[:,j],.9))))
parts=[]
for kind in ('P0','M','C','Q','G','S'):
 for task in (6,7):
  cs=[js(S/'metadata/cases'/('%s_task%d_fold%d_b50.json'%(kind,task,f))) for f in range(5)]
  parts.append(dict(kind=kind,task=task,selected_epochs=[c['selection']['selected_epoch'] for c in cs],mean_per_fold_not_pooled={p:{tar:{m:float(np.mean([c['partitions'][p]['normal'][tar]['all'][m] for c in cs])) for m in ('median_mm','p90_mm')} for tar in ('object_eef','object_goal')} for p in ('train','validation','test')}))
cost=audit['actual_cost'];summary={p:sum(r['seconds'] for r in cost['fits_detail'] if r['phase']==p) for p in ('readout','spatial')}
out=dict(status='PASS_LOCAL_PAIRED_COMPLETE_COMPARISON',comparisons=rows,partitions=parts,fit_seconds=summary,source_final_audit_sha256=hashlib.sha256((F/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),limitations=['Already inspected demos, exploratory intervals without multiplicity correction.','Relative-coordinate errors do not establish semantic query localization.','Higher patch training error prevents attribution solely to heldout overfitting.','Same encoder and query budget: this does not establish CLIP as unique bottleneck.'])
(F/'PAIRED_COMPARISON_AND_GENERALIZATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({'comparisons':rows,'fit_seconds':summary},indent=2))
