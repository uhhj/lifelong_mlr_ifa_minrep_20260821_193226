import csv,math,numpy as np
from scipy.stats import pearsonr,spearmanr
from .common import *
def rows(name):return list(csv.DictReader((OUT/name).open()))
def num(rs,key):return np.array([float(r[key]) for r in rs])
def corr(a,b,method='spearman'):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float((spearmanr if method=='spearman' else pearsonr)(a,b)[0])
def dist(a):
 a=np.asarray(a);return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),quantiles={str(q):float(np.quantile(a,q)) for q in [0,.1,.25,.5,.75,.9,1]}) if len(a) else dict(n=0)
def main():
 p=read(OUT/'PROTOCOL.json');n=read(OUT/'GN_NUMERICAL_CHECK.json');h1=rows('H1_SENSITIVITY.csv');primary=[r for r in h1 if float(r['scale'])==1.]
 out={};scales=[]
 for scale in p['scales']:
  rs=[r for r in h1 if float(r['scale'])==scale];med={kind:float(np.median(num([r for r in rs if r['direction_type']==kind],'measured_change'))) for kind in ['GN_TOP','RANDOM_ORTH_TOP8','COMPLEMENT_TOP32']}
  scales.append(dict(scale=scale,spearman=corr(num(rs,'predicted_change'),num(rs,'measured_change')),pearson=corr(num(rs,'predicted_change'),num(rs,'measured_change'),'pearson'),medians=med,top_random_ratio=med['GN_TOP']/med['RANDOM_ORTH_TOP8'],top_complement_ratio=med['GN_TOP']/med['COMPLEMENT_TOP32']))
 orders=[]
 for scale in [.25,.5]:orders.append(corr(num([r for r in h1 if float(r['scale'])==scale],'measured_change'),num(primary,'measured_change')))
 a=scales[-1];support=a['spearman'] is not None and a['spearman']>.6 and a['top_random_ratio']>=2 and a['top_complement_ratio']>=2 and min(orders)>=.9
 label='FUNCTIONAL_SENSITIVITY_SUPPORTED' if support else ('FUNCTIONAL_SENSITIVITY_NOT_SUPPORTED' if a['spearman'] is None or a['spearman']<=0 or a['top_random_ratio']<=1 else 'FUNCTIONAL_SENSITIVITY_INCONCLUSIVE')
 H1=dict(status=label,by_scale=scales,epsilon_order_spearman=orders,build_to_probe_prediction=True,limitation='Random directions in millions of dimensions provide a weak sensitivity baseline; no claim of task-specific superiority over all structured spaces.')
 write(OUT/'H1_REPORT.json',H1)
 lam=np.array(n['eigenvalues']);energy=np.cumsum(lam)/n['full_trace'];ranks={}
 for level in [.8,.9,.95]:
  ix=np.flatnonzero(energy>=level);ranks['r'+str(int(level*100))]=int(ix[0]+1) if len(ix) else '>32 (not identified)'
 pairs=rows('H2_WINDOW_REDUNDANCY.csv');stats={kind:dist(num([r for r in pairs if r['kind']==kind],'cosine')) for kind in ['ADJACENT','SAME_FAR','CROSS_TRAJECTORY']}
 demos=sorted(set(r['trajectory_i'] for r in pairs)|set(r['trajectory_j'] for r in pairs));demo_stats=[]
 for demo in demos:
  adjacent=[r for r in pairs if r['kind']=='ADJACENT' and r['trajectory_i']==demo]
  cross=[r for r in pairs if r['kind']=='CROSS_TRAJECTORY' and demo in (r['trajectory_i'],r['trajectory_j'])]
  if adjacent and cross:demo_stats.append(dict(trajectory=demo,adjacent_n=len(adjacent),cross_n=len(cross),mean_gap=float(num(adjacent,'cosine').mean()-num(cross,'cosine').mean())))
 gap=stats['ADJACENT'].get('median',0)-stats['CROSS_TRAJECTORY'].get('median',0)
 concentrated=isinstance(ranks['r90'],int) and ranks['r90']<=32
 adjacent_support=gap>=.01 and len(demo_stats)==4 and all(d['mean_gap']>0 for d in demo_stats)
 label='LOW_RANK_REDUNDANCY_SUPPORTED' if concentrated and adjacent_support else 'LOW_RANK_REDUNDANCY_INCONCLUSIVE'
 if energy[-1]<.5 and gap<=0:label='LOW_RANK_REDUNDANCY_NOT_SUPPORTED'
 H2=dict(status=label,N=128,rank_bound=min(128*7,read(OUT/'EXPERIMENT_CONTEXT.json')['P']),full_trace=n['full_trace'],captured_energy={str(k):float(energy[k-1]) for k in [1,2,4,8,16,32]},**ranks,
  pair_distributions=stats,per_demo=demo_stats,adjacent_minus_cross_median=gap,concentrated=concentrated,adjacency_supported=adjacent_support,
  limitation='Pairs share windows and trajectories, not independent observations. Squared fingerprints lose signs and output orientation. Four build trajectories only; gap is an engineering effect-size check, not a population significance test.')
 write(OUT/'H2_REPORT.json',H2)
 h3=rows('H3_DIRECTION_PERTURBATION.csv');metric={};by_scale=[]
 def measure(rs):
  certain=[r for r in rs if r['label']!='WEAK'];truth=num(certain,'measured_old_change');pred=num(certain,'pred_old_change');tol=p['H3']['loss_effect_tolerance']
  signs=(np.abs(truth)>tol)&(np.sign(truth)==np.sign(pred));classstats={}
  for label in ['SHARED','CONFLICT']:
   sub=[r for r in certain if r['label']==label];old=num(sub,'measured_old_change');new=num(sub,'measured_new_change')
   success=(old<=tol)&(new<=tol) if label=='SHARED' else (old>tol)&(new<-tol)
   classstats[label]=dict(n=len(sub),joint_supported_count=int(success.sum()),joint_fraction=float(success.mean()) if len(sub) else None,mean_old_change=float(old.mean()) if len(sub) else None,mean_new_change=float(new.mean()) if len(sub) else None)
  return dict(total_directions=len(rs),nonweak=len(certain),coverage=len(certain)/len(rs),weak=len(rs)-len(certain),old_sign_accuracy=float(signs.mean()) if len(certain) else None,old_sign_correct=int(signs.sum()),old_effect_unresolved=int((abs(truth)<=tol).sum()),
   spearman=corr(pred,truth),pearson=corr(pred,truth,'pearson'),matched_probe_spearman=corr(num(certain,'matched_probe_pred_old'),truth),matched_probe_pearson=corr(num(certain,'matched_probe_pred_old'),truth,'pearson'),classes=classstats)
 for scale in p['scales']:
  m={kind:measure([r for r in h3 if r['subspace']==kind and float(r['scale'])==scale]) for kind in ['GN','RANDOM']}
  by_scale.append(dict(scale=scale,**m))
  if scale==1.:metric=m
 g=metric['GN'];r=metric['RANDOM'];accuracy_adv=None if g['old_sign_accuracy'] is None or r['old_sign_accuracy'] is None else g['old_sign_accuracy']-r['old_sign_accuracy'];spearman_adv=None if g['spearman'] is None or r['spearman'] is None else g['spearman']-r['spearman']
 conditions=dict(nonweak=g['nonweak']>=8 and r['nonweak']>=8,classes=all(v['n']>=3 for v in g['classes'].values()),accuracy=g['old_sign_accuracy'] is not None and g['old_sign_accuracy']>=.75,
  correlation=g['spearman'] is not None and g['spearman']>.6,joint_behavior=all(v['joint_fraction'] is not None and v['joint_fraction']>=.7 for v in g['classes'].values()),
  advantage_accuracy=accuracy_adv is not None and accuracy_adv>=.1,advantage_correlation=spearman_adv is not None and spearman_adv>=.1)
 if all(conditions.values()):label='SIGNED_DIRECTION_RELATION_SUPPORTED'
 elif g['nonweak']>=8 and (g['old_sign_accuracy']<=.5 or (g['spearman'] is not None and g['spearman']<=0)):label='SIGNED_DIRECTION_RELATION_NOT_SUPPORTED'
 else:label='SIGNED_DIRECTION_RELATION_INCONCLUSIVE'
 H3=dict(status=label,by_scale=by_scale,primary=metric,conditions=conditions,accuracy_advantage=accuracy_adv,spearman_advantage=spearman_adv,
  raw_baseline=read(OUT/'RAW_GRADIENT_BASELINE.json'),limitations=['All effects are head-only local GMM NLL changes, not AdamW or full training performance.','Same-probe derivative calibration does not validate cross-trajectory labels.','One task pair and development seed; 16 directions are not independent task samples.','Numerical-neutral measured effects stay in sign-accuracy denominator.'])
 write(OUT/'H3_REPORT.json',H3);write(OUT/'RANDOM_SUBSPACE_BASELINE.json',dict(rank=32,seed=2026092305,first_directions=16,matched_perturbation_norm=True,results=r,source='same private GPU generator after orthogonal QR; no task loss used to build basis'))
 supported=all(x['status'].endswith('_SUPPORTED') and not x['status'].endswith('_NOT_SUPPORTED') for x in [H1,H2,H3])
 failed=any(x['status'].endswith('_NOT_SUPPORTED') for x in [H1,H2,H3])
 summary=dict(H1=H1['status'],H2=H2['status'],H3=H3['status'],result='PASS' if supported else ('FAIL' if failed else 'INCONCLUSIVE'),gate='GN_FUNCTIONAL_GEOMETRY_GATE_PASS' if supported else 'GN_FUNCTIONAL_GEOMETRY_GATE_NOT_PASSED',training_active=False,formal_solver_modified=False,formal_training_resumed=False,optimizer_steps_executed=0)
 write(OUT/'SUMMARY.json',summary)
 report='''# GN functional geometry: local mechanism study\n\n'''
 report+=f"Overall: **{summary['result']}**. {summary['gate']}. No formal model, optimizer, buffer or solver was changed.\n\n"
 report+='## Frozen scope\n\nOriginal MLR seed45, task8 ep99 before task9; terminal expected action of five-mode GMM, policy head only. Decoder stays eval/frozen. Float32 latents and float64 diagnostic head for every branch. Expert GMM NLL is unchanged. Build, old relation-gradient and held-out old evaluation trajectories are disjoint; new gradient/probe trajectories are disjoint. Six historical native AdamW states provide a reconstructed pre-rounding median head displacement; this is not a stored bitwise step or a same-checkpoint step.\n\n'
 report+=f"## 1. Are top directions sensitive?\n\n{H1['status']}. Primary Spearman={a['spearman']:.6g}; top/random median ratio={a['top_random_ratio']:.6g}; top/complement={a['top_complement_ratio']:.6g}. Scale-order correlations={orders}. Random directions in a 2.7-million-dimensional space are a weak baseline.\n\n"
 report+=f"## 2. Is the geometry concentrated?\n\n{H2['status']}. Full trace={n['full_trace']:.9g}; top32 energy={energy[-1]:.6g}; r80/r90/r95={ranks}. The denominator is the exact streamed full Jacobian trace, not the truncated eigenspectrum. Unresolved ranks are reported as >32, not guessed.\n\n"
 report+=f"## 3. Are adjacent windows more redundant?\n\nAdjacent-minus-cross median similarity={gap:.6g}. Distributions={stats}. Per-trajectory checks={demo_stats}. Adjacent windows overlap by 7/8. Overlap association does not establish a causal origin of low rank.\n\n"
 report+=f"## 4. Do signs predict held-out harm/help?\n\n{H3['status']}. GN primary={g}. Labels were computed using separate trajectory groups. Matched-probe calibration is reported separately. Weak and unresolvable effects are not declared safe.\n\n"
 report+=f"## 5. Does GN beat random/raw gradients?\n\nRandom primary={r}. GN-minus-random accuracy={accuracy_adv}; correlation={spearman_adv}. Raw full-gradient baseline={H3['raw_baseline']}. A single checkpoint does not identify statistical predictive superiority over a whole-update cosine baseline; no such claim is made.\n\n"
 report+='## Evidence limits and next action\n\nThe action-mean GN has zero sensitivity to the GMM scale branch, while expert NLL can depend strongly on it. Fingerprints retain squared energy, not signed function responses. Only one development seed/task pair was studied; no rollout, persistent learning, float32 writeback, or performance benefit was tested. Engineering thresholds were frozen before outcome measurement. No integration into MLR/IGFA is authorized by an unpassed gate.\n'
 (OUT/'FINAL_REPORT.md').write_text(report)
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(np.arange(1,33),energy);ax[0].set(xlabel='GN rank',ylabel='Fraction of full trace',ylim=(0,1))
  for kind in stats:ax[1].hist(num([r for r in pairs if r['kind']==kind],'cosine'),bins=30,density=True,histtype='step',label=kind)
  ax[1].legend(fontsize=7);ax[1].set(xlabel='Fingerprint cosine',ylabel='Density');fig.tight_layout();fig.savefig(OUT/'H2_SPECTRUM_AND_FINGERPRINTS.png',dpi=160);plt.close(fig)
 except ImportError:pass
 write(OUT/'STATUS.json',dict(stage='COMPLETE',**summary));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
