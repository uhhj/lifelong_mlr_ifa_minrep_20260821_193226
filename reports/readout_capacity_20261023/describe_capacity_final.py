"""Final scalar-only capacity contrast against the accepted 512 baseline. No model calls."""
from pathlib import Path
import sys,json,hashlib,statistics
import numpy as np
R=Path(__file__).resolve().parent;F=R/'final_A_CAPACITY_20261023';S=Path(sys.argv[1]);E=Path(sys.argv[2])
B=R.parent/'20261023_readout_budget';BF=B/'final_A_BUDGET_20261023';BS=B/'snapshot_20261023_171206';BE=B/'final_scalars_readonly_20261023_171248'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=js(F/'LOCAL_FINAL_AUDIT.json');m=js(E/'EVIDENCE.json');ba=js(BF/'LOCAL_FINAL_AUDIT.json');bm=js(BE/'EVIDENCE.json')
assert a['status']==ba['status']=='PASS_LOCAL_COMPLETE_A'
assert a['scalar_manifest_sha256']==sha(E/'EVIDENCE.json') and ba['scalar_manifest_sha256']==sha(BE/'EVIDENCE.json')
for p,h in js(R/'PARENT_LOCAL_ACCEPTANCE.json')['files'].items():assert sha(Path(p))==h
parts=[];pooled=[];paired=[]
for width,snap,scalar,manifest in [(512,BS,BE,bm),(128,S,E,m)]:
 res=js(snap/'metadata'/('BUDGET_RESULTS.json' if width==512 else 'CAPACITY_RESULTS.json'))
 for task in (6,7):
  for kind in ('P0','G','S'):
   for budget in (50,100,200):
    cs=[js(snap/'metadata/cases'/('%s_task%d_fold%d_b%d.json'%(kind,task,f,budget))) for f in range(5)]
    parts.append(dict(temporal_ffn=width,task=task,kind=kind,budget=budget,selected_epochs=[c['selection']['selected_epoch'] for c in cs],mean_per_fold_not_pooled={part:{target:{metric:statistics.mean(c['partitions'][part]['normal'][target]['all'][metric] for c in cs) for metric in ('median_mm','p90_mm')} for target in ('object_eef','object_goal','aperture')} for part in ('train','validation','test')}))
    for target in ('object_eef','object_goal','aperture'):
     v=res['summaries']['%s_b%d_task%d'%(kind,budget,task)]['normal'][target]['all'];pooled.append(dict(temporal_ffn=width,task=task,kind=kind,budget=budget,target=target,median_mm=v['median_mm'],p90_mm=v['p90_mm']))
for task in (6,7):
 for kind in ('P0','G','S'):
  for budget in (50,100,200):
   name='%s_b%d_task%d'%(kind,budget,task);x=js(BE/(name+'.json'));y=js(E/(name+'.json'))
   assert sha(BE/(name+'.json'))==bm['case_files'][name]['sha256'] and sha(E/(name+'.json'))==m['case_files'][name]['sha256']
   assert x['ids']==y['ids'] and x['demo_ids']==y['demo_ids'] and x['dynamic']==y['dynamic']
   demo=np.array(x['demo_ids']);xe=np.array(x['modes']['normal']['error_mm']);ye=np.array(y['modes']['normal']['error_mm'])
   for j,target in enumerate(('object_eef','object_goal','aperture')):
    ds=sorted(set(demo));delta=np.array([np.median(xe[demo==d,j])-np.median(ye[demo==d,j]) for d in ds]);rng=np.random.RandomState(410023);ci=np.quantile(delta[rng.randint(0,len(ds),(10000,len(ds)))].mean(1),[.025,.975])
    paired.append(dict(task=task,kind=kind,budget=budget,target=target,contrast='FFN512_minus_FFN128',positive_means_smaller_better=True,mean_paired_demo_median_improvement_mm=float(delta.mean()),exploratory_ci95_mm=ci.tolist(),per_demo_difference_mm=dict(zip(ds,delta.tolist()))))
costs={}
for width,manifest in [(512,bm),(128,m)]:
 rows=manifest['actual_cost']['fits_detail'];assert len(rows)==30
 costs[width]={k:dict(fits=sum(r['kind']==k for r in rows),seconds=sum(r['seconds'] for r in rows if r['kind']==k),parameters=next(r['parameters'] for r in rows if r['kind']==k),updates=sum(r['updates'] for r in rows if r['kind']==k)) for k in ('P0','G','S')}
ad=js(S/'metadata/CAPACITY_ADMISSION.json');assert ad['optimizer_calls']==9 and ad['baseline_untrained_forwards']==6 and all(v['state_exact'] and v['rng_exact'] and v['output_exact'] for v in ad['baseline_512_equivalence'])
assert [costs[128][k]['parameters'] for k in ('P0','G','S')]==[483335,400775,400775]
assert [costs[512][k]['parameters']-costs[128][k]['parameters'] for k in ('P0','G','S')]==[197376]*3
out=dict(status='PASS_COMPLETE_LOCAL_DESCRIPTIVE_CAPACITY_COMPARISON',gate_budget=200,selected=a['selected'],pooled_test=pooled,partitions=parts,paired_demo_comparisons=paired,compute=costs,own_admission_optimizer_calls=9,failed_version_isolated_optimizer_calls=9,own_baseline_equivalence_forwards=6,failed_version_equivalence_forwards=6,source_final_audit_sha256=sha(F/'LOCAL_FINAL_AUDIT.json'),baseline_final_audit_sha256=sha(BF/'LOCAL_FINAL_AUDIT.json'),limitations=['Repeated demonstration reuse is exploratory; not untouched confirmatory test data','50/100 descriptive only, gate uses validation-selected best within200','Only the specified temporal FFN width changes; not a universal parameter-count effect','No claim of unique CLIP/Temporal bottleneck or closed-loop benefit; no environment trials','This is the final automatic capacity value; failure ends diagnostic monitor, no automatic scan'])
(F/'CAPACITY_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
closure=dict(status='PASS_COMPLETE_LOCAL_CLOSURE',scientific_decision=a['scientific_decision'],selected=a['selected'],complete_fits=30,epochs=6000,optimizer_updates=49800,own_admission_updates=9,failed_admission_updates=9,strict_CP_count=len(m['checkpoint_sha256']),scalar_cases=len(m['case_files']),scalar_error_rows=a['scalar_error_rows_recomputed'],snapshot_sha256=sha(S/'EVIDENCE.json'),final_audit_sha256=sha(F/'LOCAL_FINAL_AUDIT.json'),scalar_manifest_sha256=sha(E/'EVIDENCE.json'),scalar_audit_source_sha256=sha(R/'final_scalars_readonly.py'),diagnostics_sha256=sha(F/'CAPACITY_DIAGNOSTICS.json'),parent_acceptance_sha256=sha(R/'PARENT_LOCAL_ACCEPTANCE.json'),engineering_failure_sha256=sha(R/'snapshot_20261023_173025/EVIDENCE.json'),server_exit='0',private_parent_patch_SHA_verified_by_scalar_auditor=True,no_own_process_verified_by_scalar_auditor=True,policy_updates=0,environments=0)
(F/'CLOSURE_INDEX.json').write_text(json.dumps(closure,indent=2),encoding='utf8')
print(json.dumps(dict(status=out['status'],selected=a['selected'],compute=costs),indent=2))
