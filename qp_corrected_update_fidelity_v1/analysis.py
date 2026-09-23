"""Start-clustered, frozen Phase 1 gates. Window rows are never replicates."""
import numpy as np,pandas as pd
from actual_update_fidelity_v1.analysis import metrics,clean,METRICS
from .common import *

def stats(a):
 a=pd.Series(a,dtype=float).dropna()
 return dict(n=len(a),median=a.median(),q25=a.quantile(.25),q75=a.quantile(.75),mean=a.mean(),std=a.std(ddof=1))

def bootstrap(frame,reps=10000,seed=2026092401):
 groups=[d.damage_difference.to_numpy() for _,d in frame.groupby(['seed','transition'])]
 if any(not np.isfinite(a).all() for a in groups) or len(groups)!=6:return [None,None]
 rng=np.random.RandomState(seed);out=np.empty(reps)
 for i in range(reps):out[i]=np.median(np.concatenate([rng.choice(a,len(a),replace=True) for a in groups]))
 return np.quantile(out,[.025,.975]).tolist()

def fidelity_gate(frame):
 limits={'spearman':.9,'sign_accuracy':.9,'harm_recall':.8,'harm_precision':.8};checks={}
 for key,limit in limits.items():checks[key]=bool(frame[key].notna().mean()>=.5 and frame[key].median()>=limit)
 checks['decisive_coverage']=bool(frame.decisive_fraction.median()>=.5)
 return dict(checks=checks,passed=all(checks.values()))

def start_summary(old,new,solver):
 row={k:solver[k] for k in ['start_id','seed']};row.update(transition=old.transition.iloc[0],stage=old.stage.iloc[0],solver_status=solver['status'],tau=float(old.tau_zero.mean()))
 a=old.measured_delta_d0.to_numpy();b=old.measured_delta_dfull.to_numpy();tau=old.tau_zero.to_numpy();valid=np.isfinite(b).all()
 for arm,y in [('A',a),('B',b)]:
  vals=dict(mean_delta=np.mean(y),median_delta=np.median(y),harmful_fraction=np.mean(y>tau),p90=np.quantile(y,.9),max_delta=np.max(y),positive_damage=np.mean(np.maximum(y,0)))
  for k,v in vals.items():row[k+'_'+arm]=v if arm=='A' or valid else np.nan
 ha=a>tau;hb=b>tau;row.update(damage_difference=row['positive_damage_B']-row['positive_damage_A'],mean_loss_difference=np.mean(b-a),fraction_difference=np.mean(hb)-np.mean(ha) if valid else np.nan,native_harm=int(ha.sum()),native_nonharm=int((~ha).sum()))
 row.update(prevented=int((ha&~hb).sum()) if valid else np.nan,introduced=int((~ha&hb).sum()) if valid else np.nan)
 row['prevented_rate']=row['prevented']/ha.sum() if ha.any() else np.nan;row['introduced_rate']=row['introduced']/(~ha).sum() if (~ha).any() else np.nan
 na=float(new.measured_delta_d0.mean());nb=float(new.measured_delta_dfull.mean());nt=float(new.tau_zero.mean());eligible=na < -nt
 row.update(new_delta_A=na,new_delta_B=nb,new_tau=nt,plasticity_cost=nb-na,eligible_new=eligible,retention=nb/na if eligible and valid else np.nan,reversal=float(nb>=-nt) if eligible and valid else np.nan,strict_harm_reversal=float(nb>nt) if eligible and valid else np.nan)
 for k in ['native_norm','corrected_norm','correction_ratio','norm_ratio','cosine','solve_seconds','reference_seconds','seconds']:row[k]=solver.get(k)
 return row

def summarize(partial=False):
 starts=read(OUT/'START_MANIFEST.json')['starts'];records=[];raw={k:[] for k in ['OLD','NEW','REFERENCE']};ss=[];ff=[]
 for s in starts:
  path=OUT/'starts'/s['start_id']
  if not (path/'COMPLETE.json').exists():continue
  record=read(path/'COMPLETE.json');records.append(record)
  frames={k:pd.read_csv(path/(k+'.csv')) for k in raw}
  for k,d in frames.items():raw[k].append(d)
  old,new=frames['OLD'],frames['NEW'];assert len(old)==24 and len(new)==24
  ss.append(start_summary(old,new,record))
  for arm in ['d0','dfull']:
   if not np.isfinite(old['measured_delta_'+arm]).all():v={k:np.nan for k in METRICS}
   else:v=metrics(old['h_dot_'+arm],old['measured_delta_'+arm],old.tau_zero)
   ff.append(dict(start_id=s['start_id'],seed=s['seed'],transition=s['transition'],arm=arm,**v))
 if not partial:assert len(records)==len(starts)==120
 assert records
 m=pd.DataFrame(ss);f=pd.DataFrame(ff);table(OUT/'FULL_QP_SOLVER_DIAGNOSTICS.csv',records)
 m.to_csv(OUT/'START_LEVEL_METRICS.csv',index=False);f.to_csv(OUT/'POST_QP_FIDELITY.csv',index=False)
 for k,name in [('OLD','OLD_PROBE_RESULTS'),('NEW','NEW_PROBE_RESULTS'),('REFERENCE','REFERENCE_RESULTS')]:pd.concat(raw[k],ignore_index=True).to_csv(OUT/(name+'.csv'),index=False)
 cells=[]
 for (seed,transition),d in m.groupby(['seed','transition']):
  c=dict(seed=int(seed),transition=transition,starts=len(d),failed=int((d.solver_status!='PASS').sum()))
  for key in m.select_dtypes(include=[np.number]).columns:
   if key=='seed':continue
   for suffix,v in stats(d[key]).items():c[key+'_'+suffix]=v
  cells.append(c)
 ct=pd.DataFrame(cells);ct.to_csv(OUT/'SEED_TRANSITION_METRICS.csv',index=False)
 complete=len(records)==120;failed=int((m.solver_status!='PASS').sum());ci=bootstrap(m) if complete and not failed else [None,None]
 med=float(m.damage_difference.median());negative=int((ct.damage_difference_median<0).sum());reverse=(ct.damage_difference_median> .1*ct.positive_damage_A_median+ct.tau_mean)
 checks=dict(all_starts_solved=complete and failed==0,negative_median=med<0,cells=negative>=5,no_strong_reverse=not bool(reverse.any()),CI=ci[1] is not None and ci[1]<0,harmful_fraction=float(m.fraction_difference.median())<0,introduced_not_offsetting=float(m.introduced.sum())<=.5*float(m.prevented.sum()) and float(m.prevented.sum())>0)
 protection='SUPPORTED' if all(checks.values()) else ('NOT_SUPPORTED' if (med>=0 or int((ct.damage_difference_median>0).sum())>=4) and not failed and complete else 'INCONCLUSIVE')
 p=clean(dict(classification=protection,primary='mean positive actual old delta per start; paired median B-A',median_damage_difference=med,cluster_bootstrap95=ci,checks=checks,cells=cells,prevented_count=m.prevented.sum(),introduced_count=m.introduced.sum(),prevented_rate_start_summary=stats(m.prevented_rate),introduced_rate_start_summary=stats(m.introduced_rate),solver_failed=failed,all_manifest_starts=len(starts)))
 write(OUT/'QP_PROTECTION_REPORT.json',p)
 fb=f[f.arm=='dfull'];ft=[]
 for (seed,transition),d in fb.groupby(['seed','transition']):ft.append(dict(seed=int(seed),transition=transition,**fidelity_gate(d),metrics={k:stats(d[k]) for k in METRICS}))
 globalf=fidelity_gate(fb);ok=globalf['passed'] and sum(r['passed'] for r in ft)>=5 and complete and failed==0
 fidelity='SUPPORTED' if ok else ('NOT_SUPPORTED' if complete and failed==0 and (fb.spearman.median()<.6 or fb.sign_accuracy.median()<.7) else 'INCONCLUSIVE')
 paired=f.pivot(index='start_id',columns='arm',values=METRICS);diff={k:stats(paired[k]['dfull']-paired[k]['d0']) for k in METRICS}
 write(OUT/'POST_QP_FIDELITY_REPORT.json',clean(dict(classification=fidelity,overall=globalf,metrics={k:stats(fb[k]) for k in METRICS},cells=ft,paired_change_from_d0=diff)))
 eligible=m[m.eligible_new & m.new_delta_B.notna()];ret=float(eligible.retention.median());rev=float(eligible.reversal.mean())
 plasticity='UNCLASSIFIABLE' if not len(eligible) else ('LOW_COST' if ret>=.9 and rev<.1 else ('SEVERE_COST' if ret<.5 or rev>.3 else 'MATERIAL_COST'))
 pc=[]
 for key,d in m.groupby(['seed','transition']):
  e=d[d.eligible_new & d.new_delta_B.notna()];pc.append(dict(seed=int(key[0]),transition=key[1],eligible=len(e),cost=stats(d.plasticity_cost),retention=stats(e.retention),reversal=e.reversal.mean(),strict_harm_reversal=e.strict_harm_reversal.mean()))
 write(OUT/'PLASTICITY_REPORT.json',clean(dict(classification=plasticity,eligible_native_improving=len(eligible),retention_median=ret,reversal=rev,cost=stats(m.plasticity_cost),cells=pc)))
 # Explicit stratification of reference and truly held-out trajectories, not only windows.
 generalization=[]
 for role in raw:
  d=pd.concat(raw[role]);sets=[('all',d)]
  if role=='OLD':sets += [('trajectory_disjoint',d[~d.reference_trajectory_overlap]),('same_reference_trajectory',d[d.reference_trajectory_overlap])]
  for label,sub in sets:
   grouped=sub.groupby('start_id').protection_gain.mean();generalization.append(dict(role=role,subset=label,windows=len(sub),starts=len(grouped),mean_loss_gain_across_starts=stats(grouped)))
 write(OUT/'REFERENCE_VS_PROBE_REPORT.json',clean(dict(groups=generalization,independence='No exact window overlap. Most preserved old probes share a reference trajectory; this is not unseen-trajectory confirmation.')))
 gate=protection==fidelity=='SUPPORTED' and complete and failed==0
 mechanism='SUPPORTED' if gate and plasticity=='LOW_COST' else ('CONDITIONAL' if gate else 'NOT_ESTABLISHED')
 summary=clean(dict(status='COMPLETE' if complete else 'PARTIAL',starts=len(records),solver_failed=failed,QP_PROTECTION=protection,POST_QP_FIRST_ORDER_FIDELITY=fidelity,PLASTICITY_COST=plasticity,SAMPLE_QP_MECHANISM=mechanism,phase2_gate=gate,phase2_status='PENDING' if gate else 'NOT_RUN_GATE_NOT_PASSED',formal_training_resumed=False))
 if (OUT/'GN_WORKSET_VALUE_REPORT.json').exists():summary.update(GN_WORKSET_VALUE=read(OUT/'GN_WORKSET_VALUE_REPORT.json')['classification'],phase2_status='COMPLETE')
 write(OUT/'SUMMARY.json',summary)
 report=['# QP corrected-update local mechanism', '', 'Formal training remains stopped. This is a paired single-step diagnostic.', '', '```json',__import__('json').dumps(summary,indent=2),'```','',f'Primary paired median positive-damage difference: {med}; start-cluster 95% CI: {ci}.',f'Prevented windows: {m.prevented.sum()}; newly harmed: {m.introduced.sum()}. Rates retain their distinct denominators.',f'Native-improving starts: {len(eligible)}; median retained new progress: {ret}; reversal: {rev}.','', '|seed|transition|starts|solver failures|median old damage B-A|mean new cost|','|---|---|---:|---:|---:|---:|']
 for c in cells:report.append(f"|{c['seed']}|{c['transition']}|{c['starts']}|{c['failed']}|{c['damage_difference_median']}|{c['plasticity_cost_mean']}|")
 report += ['', '## Interpretation and required answers',f'1. Independent-window protection: {protection}. The primary outcome is positive damage, not reference feasibility.',f'2. {negative}/6 cells have negative median paired damage; all six cells are shown, including failures.', '3. Reference results and held-out windows/trajectories are separated in REFERENCE_VS_PROBE_REPORT. Exact overlap was removed before outcomes; 122/2880 prior probes replaced by a deterministic identity-only rule. Most remaining probes share reference demonstrations, limiting trajectory generalization.',f'4. Corrected-update Taylor fidelity: {fidelity}; medians and paired decline from native d0 are in POST_QP_FIDELITY_REPORT.', '5. Actual float32-written displacements, norm ratios, cosine and correction ratios are recorded per start, distinct from ideal solver correction.',f'6–7. Prevented and introduced harm counts/rates above must be considered together.',f'8. New-task cost: {plasticity}, with per-cell costs and discernibly improving-start retention.', '9. These measurements can reveal a local stability/plasticity tradeoff; they cannot explain cumulative training or rollout behavior.',f'10. Mechanism status: {mechanism}. If supported with material cost, next study is paired 5/20 steps, not full lifelong training.',f'11–16. Phase 2: {summary["phase2_status"]}; conditional gate is {gate}. No GN sharing/conflict gate is used.', '', '## Limits', '- Development seeds45/46/47; source stages cover epochs30–99, not early epochs.', '- 120 starts share carriers and probes; confidence intervals cluster by start within cells, not by window. Carrier/trajectory dependence still limits broader inference.', '- Formal v4 reference sampling is reused literally, including its historical fixed46 RNG salt. Constraints count32/64, not a fabricated64 for every start.', '- No formal training, rollout, optimizer change, KKT relaxation, fallback change, or formal asset mutation.', '- Numerical failures remain in the denominator and block Phase2; failed corrected outcomes are missing, never zero.']
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n')
 if complete:plots(m,pd.concat(raw['OLD']))
 return summary

def plots(m,old):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 specs=[('old_delta_d0_vs_dfull','mean_delta_A','mean_delta_B'),('harmful_fraction_d0_vs_dfull','harmful_fraction_A','harmful_fraction_B'),('correction_ratio_vs_protection_gain','correction_ratio','damage_difference')]
 for name,x,y in specs:
  fig,ax=plt.subplots();ax.scatter(m[x],m[y],s=12);ax.set(xlabel=x,ylabel=y,title='Each point is a start state');fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=130);plt.close(fig)
 for name,key in [('per_start_old_protection_gain','damage_difference'),('new_task_plasticity_cost','plasticity_cost')]:
  fig,ax=plt.subplots();ax.scatter(np.arange(len(m)),m[key],s=10);ax.axhline(0,color='black');ax.set(xlabel='Frozen start order',ylabel=key);fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=130);plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(old.h_dot_dfull,old.measured_delta_dfull,s=4,alpha=.3);ax.set(xlabel='h_i dot physically written d_full',ylabel='Measured old NLL change',title='Descriptive windows, not independent replicates');fig.tight_layout();fig.savefig(OUT/'predicted_vs_measured_dfull.png',dpi=130);plt.close(fig)
if __name__=='__main__':summarize()
