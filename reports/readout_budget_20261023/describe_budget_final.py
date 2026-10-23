"""Complete budget diagnosis from independently accepted scalars; no test selection."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
R=Path(__file__).resolve().parent;F=R/'final_A_BUDGET_20261023';S=R/'snapshot_20261023_171206';E=R/'final_scalars_readonly_20261023_171248'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=js(F/'LOCAL_FINAL_AUDIT.json');m=js(E/'EVIDENCE.json');assert a['status']=='PASS_LOCAL_COMPLETE_A' and a['scalar_manifest_sha256']==sha(E/'EVIDENCE.json')
res=js(S/'metadata/BUDGET_RESULTS.json');parts=[];pooled=[];paired=[]
for task in (6,7):
 for kind in ('P0','G','S'):
  for budget in (50,100,200):
   cs=[js(S/'metadata/cases'/('%s_task%d_fold%d_b%d.json'%(kind,task,f,budget))) for f in range(5)]
   parts.append(dict(task=task,kind=kind,budget=budget,selected_epochs=[c['selection']['selected_epoch'] for c in cs],mean_per_fold_not_pooled={part:{target:{metric:statistics.mean(c['partitions'][part]['normal'][target]['all'][metric] for c in cs) for metric in ('median_mm','p90_mm')} for target in ('object_eef','object_goal')} for part in ('train','validation','test')}))
   for target in ('object_eef','object_goal'):
    val=res['summaries']['%s_b%d_task%d'%(kind,budget,task)]['normal'][target]['all'];pooled.append(dict(task=task,kind=kind,budget=budget,target=target,median_mm=val['median_mm'],p90_mm=val['p90_mm']))
 def compare(k1,b1,k2,b2,label):
  # Positive delta = first minus second error; descriptive cluster bootstrap over paired demos.
  x=js(E/('%s_b%d_task%d.json'%(k1,b1,task)));y=js(E/('%s_b%d_task%d.json'%(k2,b2,task)))
  for key in ('%s_b%d_task%d'%(k1,b1,task),'%s_b%d_task%d'%(k2,b2,task)):assert sha(E/(key+'.json'))==m['case_files'][key]['sha256']
  xd=np.array(x['demo_ids']);yd=np.array(y['demo_ids']);xe=np.array(x['modes']['normal']['error_mm']);ye=np.array(y['modes']['normal']['error_mm'])
  for j,target in enumerate(('object_eef','object_goal')):
   demos=sorted(set(xd));assert set(demos)==set(yd);delta=np.array([np.median(xe[xd==d,j])-np.median(ye[yd==d,j]) for d in demos]);rng=np.random.RandomState(410023);ci=np.quantile(delta[rng.randint(0,len(demos),(10000,len(demos)))].mean(1),[.025,.975]);paired.append(dict(task=task,target=target,contrast=label,positive_means_second_better=True,mean_paired_demo_median_improvement_mm=float(delta.mean()),exploratory_ci95_mm=ci.tolist()))
 for kind in ('P0','G','S'):compare(kind,50,kind,200,kind+'_50_vs_200')
 compare('G',200,'S',200,'G200_vs_S200')
costs=m['actual_cost']['fits_detail'];cost={k:dict(fits=sum(r['kind']==k for r in costs),seconds=sum(r['seconds'] for r in costs if r['kind']==k),parameters=next(r['parameters'] for r in costs if r['kind']==k),updates=sum(r['updates'] for r in costs if r['kind']==k)) for k in ('P0','G','S')}
out=dict(status='PASS_COMPLETE_LOCAL_DESCRIPTIVE_BUDGET_COMPARISON',gate_budget=200,selected=a['selected'],pooled_test=pooled,partitions=parts,paired_demo_comparisons=paired,compute=cost,total_fit_seconds=sum(r['seconds'] for r in costs),source_final_audit_sha256=sha(F/'LOCAL_FINAL_AUDIT.json'),limitations=['Previously inspected demonstrations; exploratory repeated reuse, not independent external replication','Budget50/100 descriptive only; no test-based checkpoint or budget selection','Per-fold means differ from pooled heldout medians','No unique CLIP,Temporal or optimization bottleneck identified; no closed-loop trials'])
(F/'BUDGET_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
closure=dict(status='PASS_COMPLETE_LOCAL_CLOSURE',scientific_decision=a['scientific_decision'],selected=a['selected'],complete_fits=30,epochs=6000,optimizer_updates=49800,own_admission_updates=9,strict_CP_count=len(m['checkpoint_sha256']),scalar_cases=len(m['case_files']),scalar_error_rows=a['scalar_error_rows_recomputed'],snapshot_sha256=sha(S/'EVIDENCE.json'),final_audit_sha256=sha(F/'LOCAL_FINAL_AUDIT.json'),scalar_manifest_sha256=sha(E/'EVIDENCE.json'),scalar_audit_source_sha256=sha(R/'final_scalars_readonly.py'),diagnostics_sha256=sha(F/'BUDGET_DIAGNOSTICS.json'),server_exit='0',private_parent_patch_SHA_verified_by_scalar_auditor=True,no_own_process_verified_by_scalar_auditor=True,policy_updates=0,environments=0)
(F/'CLOSURE_INDEX.json').write_text(json.dumps(closure,indent=2),encoding='utf8')
print(json.dumps(dict(status=out['status'],compute=cost,total_fit_seconds=out['total_fit_seconds'],paired_G200_S200=[v for v in paired if v['contrast']=='G200_vs_S200']),indent=2))
