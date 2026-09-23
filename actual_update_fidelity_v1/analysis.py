import math,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from .common import *
METRICS=['spearman','pearson','sign_accuracy','harm_recall','harm_precision','slope','intercept','mae','relative_mae','decisive_fraction']
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

def metrics(pred,actual,tau):
 pred=np.asarray(pred,float);actual=np.asarray(actual,float);tau=np.asarray(tau,float)
 sp=np.where(abs(pred)<=tau,0,np.sign(pred));sy=np.where(abs(actual)<=tau,0,np.sign(actual));both=(sp!=0)&(sy!=0)
 valid=np.std(pred)>0 and np.std(actual)>0
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');sr=float(spearmanr(pred,actual)[0]) if valid else np.nan;pr=float(pearsonr(pred,actual)[0]) if valid else np.nan
 slope=float(np.dot(pred-pred.mean(),actual-actual.mean())/np.square(pred-pred.mean()).sum()) if np.std(pred)>0 else np.nan
 return dict(windows=len(pred),spearman=sr,pearson=pr,sign_accuracy=float(np.mean(sp[both]==sy[both])) if both.any() else np.nan,harm_recall=float(np.mean(sp[sy>0]>0)) if (sy>0).any() else np.nan,harm_precision=float(np.mean(sy[sp>0]>0)) if (sp>0).any() else np.nan,slope=slope,intercept=float(actual.mean()-slope*pred.mean()),mae=float(np.mean(abs(actual-pred))),relative_mae=float(np.mean(abs(actual-pred))/max(np.mean(abs(actual)),np.mean(tau))),decisive_fraction=float(both.mean()),predicted_neutral=int((sp==0).sum()),measured_neutral=int((sy==0).sum()),actual_harm=int((sy>0).sum()),predicted_harm=int((sp>0).sum()))

def summarize(partial=False):
 manifest=read(OUT/'START_MANIFEST.json');files=[OUT/'starts'/s['start_id']/'measurements.csv' for s in manifest['starts'] if (OUT/'starts'/s['start_id']/'COMPLETE.json').exists()]
 assert files,'No complete measurements'
 if not partial:assert len(files)==120,('Incomplete',len(files))
 data=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);stats=[]
 for key,df in data.groupby(['start_id','seed','transition','stage','arm','alpha'],sort=True):
  stats.append(dict(zip(['start_id','seed','transition','stage','arm','alpha'],key),**metrics(df.predicted_delta_loss,df.measured_delta_loss,df.tau_zero)))
 m=pd.DataFrame(stats);m.to_csv(OUT/'START_LEVEL_METRICS.csv',index=False)
 for arm,file in [('adamw','ACTUAL_UPDATE_FIDELITY.csv'),('raw','RAW_GRADIENT_BASELINE.csv')]:data[data.arm==arm].to_csv(OUT/file,index=False)
 aggregates=[]
 for key,df in m.groupby(['seed','transition','arm','alpha']):
  row=dict(zip(['seed','transition','arm','alpha'],key),starts=len(df))
  for k in METRICS:
   v=df[k].dropna();row.update({k+'_median':v.median(),k+'_q25':v.quantile(.25),k+'_q75':v.quantile(.75),k+'_mean':v.mean(),k+'_std':v.std(ddof=1),k+'_defined_starts':len(v)})
  aggregates.append(row)
 strat=pd.DataFrame(aggregates);strat.to_csv(OUT/'SEED_TRANSITION_METRICS.csv',index=False)
 # Across seeds: each seed has equal weight; do not pool window observations.
 seed_rows=[]
 for key,df in strat.groupby(['seed','arm','alpha']):
  seed_rows.append(dict(zip(['seed','arm','alpha'],key),**{k:df[k+'_mean'].mean() for k in METRICS}))
 seedtable=pd.DataFrame(seed_rows);seedtable.to_csv(OUT/'SEED_LEVEL_METRICS.csv',index=False)
 rules=read(OUT/'PROTOCOL.json')['reliability'];alpha_rows=[]
 for alpha in ALPHAS:
  ds=strat[(strat.arm=='adamw')&(strat.alpha==alpha)];tests=[]
  for _,row in ds.iterrows():
   checks={k:bool(row[k+'_median']>=rules['median_'+k] and row[k+'_defined_starts']>=rules['minimum_defined_start_fraction']*row.starts) for k in ['spearman','sign_accuracy','harm_recall','harm_precision']}
   checks['decisive_coverage']=bool(row.decisive_fraction_median>=rules['minimum_decisive_window_fraction'])
   tests.append(dict(seed=int(row.seed),transition=row.transition,checks=checks,reliable=all(checks.values())))
  a=m[(m.arm=='adamw')&(m.alpha==alpha)];alpha_rows.append(dict(alpha=alpha,reliable=len(tests)==6 and all(t['reliable'] for t in tests),strata=tests,median_across_starts={k:a[k].median() for k in METRICS},mean_equal_seed={k:seedtable[(seedtable.arm=='adamw')&(seedtable.alpha==alpha)][k].mean() for k in METRICS}))
 reliable=[a['alpha'] for a in alpha_rows if a['reliable']]
 if 1. in reliable:label='FIRST_ORDER_ACTUAL_UPDATE_SUPPORTED'
 elif any(a in reliable for a in (.5,.25,.125)):label='FIRST_ORDER_TRUST_REGION_SUPPORTED'
 elif reliable:label='FIRST_ORDER_LOCAL_ONLY'
 else:label='FIRST_ORDER_ACTUAL_UPDATE_NOT_ESTABLISHED'
 large=alpha_rows[0]['median_across_starts'];small=alpha_rows[-1]['median_across_starts']
 trigger=bool(not alpha_rows[0]['reliable'] and (small['spearman']-large['spearman']>=.15 or small['sign_accuracy']-large['sign_accuracy']>=.10) and small['decisive_fraction']>=.5)
 write(OUT/'ALPHA_SWEEP_REPORT.json',clean(dict(alpha_results=alpha_rows,classification=label,hvp_trigger=trigger,engineering_rules=rules,not_a_theorem=True)))
 # Paired self-direction fidelity AND raw proxy for the exact same AdamW outcomes.
 pair=m.pivot(index=['start_id','seed','transition','stage','alpha'],columns='arm',values=METRICS)
 paired=pd.DataFrame({k:pair[(k,'adamw')]-pair[(k,'raw')] for k in METRICS}).reset_index();paired.to_csv(OUT/'PAIRED_RAW_ADAMW_METRICS.csv',index=False)
 proxy=[]
 for key,df in data[data.arm=='adamw'].groupby(['start_id','seed','transition','stage','alpha']):
  r=data[(data.start_id==key[0])&(data.arm=='raw')&(data.alpha==key[-1])]
  merged=df.merge(r[['record_id','predicted_delta_loss']],on='record_id',suffixes=('','_raw'),validate='one_to_one')
  proxy.append(dict(zip(['start_id','seed','transition','stage','alpha'],key),**metrics(merged.predicted_delta_loss_raw,merged.measured_delta_loss,merged.tau_zero)))
 proxy=pd.DataFrame(proxy);proxy.to_csv(OUT/'RAW_PROXY_FOR_ACTUAL_ADAMW.csv',index=False)
 comparison=[]
 for key,df in paired[paired.alpha==1].groupby(['seed','transition']):comparison.append(dict(seed=int(key[0]),transition=key[1],**{k:df[k].median() for k in METRICS}))
 # Descriptive consistency only; no pseudo significance from2880 windows.
 support=all(r['spearman']>=.05 and r['sign_accuracy']>=0 and r['harm_recall']>=0 and r['harm_precision']>=-.02 for r in comparison)
 reverse=all(r['spearman']<=-.05 and r['sign_accuracy']<=0 for r in comparison)
 rawlabel='SUPPORTED' if support else ('NOT_SUPPORTED' if reverse else 'INCONCLUSIVE')
 write(OUT/'RAW_VS_ADAMW_REPORT.json',clean(dict(ADAMW_OVER_RAW_GRADIENT=rawlabel,paired_alpha1_strata=comparison,interpretation='Self-direction accuracy compares two different interventions. Same-target raw-proxy analysis is separately reported; raw gradient is not an AdamW displacement.',criterion='all six strata median Spearman gain>=.05 with no sign/recall decline and precision decline<=.02; otherwise descriptive inconclusive unless consistent reverse')))
 noise=[dict(start_id=read(p.parent/'COMPLETE.json')['start_id'],**read(p.parent/'COMPLETE.json')['noise']) for p in files]
 write(OUT/'NUMERICAL_NOISE_FLOOR.json',dict(rule=read(OUT/'PROTOCOL.json')['deadband'],starts=noise))
 sample='SUPPORTED' if label.endswith('UPDATE_SUPPORTED') else ('CONDITIONAL' if label in ('FIRST_ORDER_TRUST_REGION_SUPPORTED','FIRST_ORDER_LOCAL_ONLY') else 'NOT_ESTABLISHED')
 summary=clean(dict(status='PARTIAL' if partial else 'COMPLETE',starts=len(files),window_pairs=len(data[data.arm=='adamw'])//len(ALPHAS),classification=label,ADAMW_OVER_RAW_GRADIENT=rawlabel,SAMPLE_QP_MECHANISM=sample,hvp_trigger=trigger,hvp_completed=(OUT/'SECOND_ORDER_REPORT.json').exists(),formal_training_resumed=False,scope='Native d0 only; no finite-step guarantee for a different sample-QP corrected displacement. All starts are development carrier states; no rollout evidence.',limitations=read(OUT/'EXPERIMENT_CONTEXT.json')['limitations']))
 write(OUT/'SUMMARY.json',summary)
 if not partial:plots(data,m,paired)
 lines=['# Actual AdamW update fidelity',f'\nStatus: {summary["status"]}; starts={len(files)}; formal_training_resumed=false.',f'\nClassification: `{label}`. Sample-QP local mechanism: `{sample}`. AdamW-over-raw: `{rawlabel}`.','\nAll thresholds are preregistered engineering interpretation, not proof of protection.','\n|alpha|Spearman median|sign accuracy|harm recall|harm precision|decisive coverage|all strata reliable|','|---:|---:|---:|---:|---:|---:|---|']
 for a in alpha_rows:
  v=a['median_across_starts'];lines.append('|'+ '|'.join(str(x) for x in [a['alpha'],v['spearman'],v['sign_accuracy'],v['harm_recall'],v['harm_precision'],v['decisive_fraction'],a['reliable']])+'|')
 lines+=['\n1. Normal-scale prediction: see alpha1 and all six stratum gates above.',f'2. Reliable tested scales: {reliable}; no interpolation to untested alpha.',f'3. Paired actual-versus-raw conclusion: {rawlabel}; see same-target proxy separately.','4. Ranking, sign and magnitude errors are separately quantified in START_LEVEL_METRICS (Spearman, sign, slope/intercept/relative MAE).','5. Harm recall/precision include neutral errors in their respective denominators; undefined cases are not silently scored as correct.','6. Seed and transition consistency: ALPHA_SWEEP_REPORT contains every stratum; no pooled-window vote.',f'7. Sample-QP mechanism: {sample}, only for the measured native directions. This never proves non-increasing finite loss for all QP-corrected displacements.','8. If only reduced steps pass, prioritize a measured trust region/line search; if none pass, audit numerical fidelity and use actual-loss probes before stronger protection claims.',f'9. True NLL HVP conditional trigger={trigger}; completion={summary["hvp_completed"]}. No Fisher substitution.','10. Formal lifelong training remains stopped; no automatic restart.','\n## Scope limitations']+['- '+x for x in summary['limitations']]
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 return summary

def second_report(rows):
 df=pd.DataFrame(rows);r=[]
 for sid,d in df.groupby('start_id'):
  r.append(dict(start_id=sid,first_mae=float(np.mean(abs(d.first_order-d.measured_delta_loss))),second_mae=float(np.mean(abs(d.second_order-d.measured_delta_loss)))))
 write(OUT/'SECOND_ORDER_REPORT.json',dict(true_NLL_Hessian=True,starts=len(r),per_start=r,improved_starts=sum(x['second_mae']<x['first_mae'] for x in r),median_mae_difference=float(np.median([x['second_mae']-x['first_mae'] for x in r])),formal_training_resumed=False))

def plots(data,m,paired):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for key in ['spearman','sign_accuracy','harm_recall','harm_precision']:
  fig,ax=plt.subplots()
  for arm in ['adamw','raw']:
   a=m[m.arm==arm].groupby('alpha')[key];med=a.median();lo=a.quantile(.25);hi=a.quantile(.75)
   ax.plot(med.index,med.values,'o-',label=arm);ax.fill_between(med.index,lo.values,hi.values,alpha=.18)
  ax.set(xscale='log',xlabel='alpha',ylabel=key,title='Median across starts; shading = IQR');ax.legend();fig.tight_layout();fig.savefig(OUT/f'alpha_vs_{key}.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();d=data[(data.arm=='adamw')&(data.alpha==1)];ax.scatter(d.predicted_delta_loss,d.measured_delta_loss,s=4,alpha=.4);lo=min(d.predicted_delta_loss.min(),d.measured_delta_loss.min());hi=max(d.predicted_delta_loss.max(),d.measured_delta_loss.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='First order prediction',ylabel='Actual GMM NLL change',title='Descriptive windows; inferential unit = start');fig.tight_layout();fig.savefig(OUT/'predicted_vs_measured_alpha1.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();d=paired[paired.alpha==1];ax.boxplot([d[k].dropna() for k in ['spearman','sign_accuracy','harm_recall','harm_precision']],labels=['Spearman','sign','recall','precision']);ax.axhline(0,color='k');ax.set_ylabel('Paired AdamW minus raw at alpha1');fig.tight_layout();fig.savefig(OUT/'raw_vs_adamw_comparison.png',dpi=150);plt.close(fig)
if __name__=='__main__':summarize(partial=False)
