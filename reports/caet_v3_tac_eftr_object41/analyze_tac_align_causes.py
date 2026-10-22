"""Read-only local decomposition; no training, model forward, or remote access."""
import hashlib,json,datetime
from pathlib import Path
from fractions import Fraction as F
R=Path(__file__).resolve().parent
P={'TAC':R/'final_TAC_seed41_20261022_165721','ALIGN':R/'final_ALIGN_seed41_20261022_220252'}
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=js(R/'tac_align_final_comparison_20261022/COMPARISON.json')
assert c['status']=='ACCEPTED_RUNS_COMPARISON_PASS'
rates={n:{(r['stage'],r['task']):F(r[n]) for r in c['matrix']} for n in ('RA','ALIGN','TAC')}
decomp={}
for x,y in [('TAC','ALIGN'),('ALIGN','RA')]:
 a,b=rates[x],rates[y]
 vals={str(t):dict(auc_contribution_pp=float(100*(sum(a[s,t]-b[s,t] for s in range(t,10))/(10-t))/4),
                   fwt_contribution_pp=float(100*(a[t,t]-b[t,t])/4),
                   initial_delta_pp=float(100*(a[t,t]-b[t,t])),
                   final_delta_pp=float(100*(a[9,t]-b[9,t]))) for t in range(6,10)}
 assert abs(sum(v['auc_contribution_pp'] for v in vals.values())-c['comparisons'][x+'_minus_'+y]['AUC']['percent'])<1e-10
 decomp[x+'_minus_'+y]=vals
stats={};checked=0
manifests={n:js(p/'PACKAGE_MANIFEST.json')['entries'] for n,p in P.items()}
for t in range(6,10):
 stage=[]
 for ep in range(100):
  rel=f'current/ra_coreset/auxiliary/task{t}_epoch{ep:02d}.json'
  pair={}
  for n,p in P.items():
   assert sha(p/rel)==manifests[n][rel]['sha256']
   pair[n]=js(p/rel)
  assert len(pair['TAC'])==len(pair['ALIGN'])
  for a,b in zip(pair['TAC'],pair['ALIGN']):
   assert all(a[k]==b[k] for k in ('step','batch_ids','bc_valid','bc_invalid','bc_current_valid'))
   assert not any('gradient' in k or 'grad_norm' in k for k in a)
   checked+=1
   stage.append(a)
 md={n:js(p/f'current/ra_coreset/dataset_manifests/task{t}.json') for n,p in P.items()}
 for a,b in zip(md['TAC']['rows'],md['ALIGN']['rows']):
  assert all(a[k]==b[k] for k in ('index','sample','task','demo','u','sha256','bc_valid','effect_valid'))
 positive=md['TAC']['positive'];negative=md['TAC']['negative']
 anchors=[i for i in positive if positive[i] and negative.get(i)]
 unique={(md['TAC']['rows'][int(i)]['task'],md['TAC']['rows'][int(i)]['demo'],md['TAC']['rows'][int(i)]['u']) for i in anchors}
 current={ (r['task'],r['demo'],r['u']) for r in md['TAC']['rows'] if r['task']==t and r['effect_valid']}
 def mean(k,rows):
  values=[r[k] for r in rows if r.get(k) is not None]
  return sum(values)/len(values) if values else None
 stats[str(t)]=dict(main_updates=len(stage),effect_samples=sum(s['effect_samples'] for s in stage),
                   contrast_anchor_exposures=sum(s['cross_anchors'] for s in stage),
                   unique_supported_moments=len(unique),current_unique_effect_valid_moments=len(current),
                   support_fraction_of_unique_effect_valid_current=len(unique)/len(current),
                   head_updates=sum(s['head_updates'] for s in stage),
                   effect_loss_mean=mean('effect_loss',stage),cross_loss_mean_including_zero_batches=mean('cross_loss',stage))
 assert stats[str(t)]['contrast_anchor_exposures']==len(unique)*100
assert checked==105500
nbt={n:{str(t):float(100*(r[t,t]-sum(r[s,t] for s in range(t+1,10))/(9-t))) for t in range(6,9)} for n,r in rates.items()}
# Explanatory common-anchor arithmetic only; this is not a replacement reported metric.
common_anchor_tac=sum(rates['ALIGN'][t,t]-sum(rates['TAC'][s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3
out=dict(status='LOCAL_CAUSAL_LIMITS_AND_DECOMPOSITION_PASS',observed_at=datetime.datetime.now().astimezone().isoformat(),
         matched_batch_and_mask_steps=checked,decomposition=decomp,stage_auxiliary=stats,nbt_by_task_pp=nbt,
         illustrative_TAC_NBT_with_ALIGN_anchors_percent=float(100*common_anchor_tac),
         common_anchor_note='Arithmetic illustration only, not an alternative official metric or causal counterfactual.',
         formal_gradient_conflict_measurements_available=False,
         causal_limits=['One seed, 20 episodes per cell; no significance claim.',
                       'TAC vs ALIGN isolates the joint auxiliary implementation, not effect versus contrast independently.',
                       'Task7 has zero contrast, but starts from a different Task6 model; its deficit cannot be assigned solely to Task7 regression.',
                       'Loss scalars do not establish gradient dominance; no frozen-policy forward or gradients were run.'],
         input_comparison_sha256=sha(R/'tac_align_final_comparison_20261022/COMPARISON.json'),
         source_review_sha256={str(p.relative_to(R)):sha(p) for p in [R/'PROTOCOL.md',R/'additions/caet_v3_exp/observe.py',R/'additions/caet_v3_exp/model.py',R/'additions/caet_v3_exp/dataset.py']},
         training_updates=0,model_forwards=0,environment_steps=0,server_writes=0)
dest=R/'tac_align_cause_analysis_20261022';dest.mkdir(exist_ok=True)
(dest/'LOCAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
