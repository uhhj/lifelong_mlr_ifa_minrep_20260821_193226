"""Frozen grouped prediction analysis: no hyperparameter or threshold search."""
import argparse,collections,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,pearsonr
from .common import *

def corr(x,y,kind='spearman'):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    v=(spearmanr(x,y) if kind=='spearman' else pearsonr(x,y))[0]
    return float(v) if np.isfinite(v) else None
def no_leak(train,test):
    return train['start_id']!=test['start_id'] and not ({train['new_trajectory'],train['old_trajectory']}&{test['new_trajectory'],test['old_trajectory']})
def predict(train,test,features):
    X=np.array([[r[k] for k in features] for r in train],float);y=np.array([r['old_delta'] for r in train]);mu=X.mean(0);sd=np.maximum(X.std(0),1e-12);X=(X-mu)/sd
    ymean=y.mean();coef=np.linalg.solve(X.T@X+np.eye(len(features)),X.T@(y-ymean));return float(((np.array([test[k] for k in features])-mu)/sd)@coef+ymean)
def boot(values,seed=51427):
    a=np.array(values,float)
    if len(a)<2:return None
    rng=np.random.RandomState(seed);v=np.mean(a[rng.randint(0,len(a),(2000,len(a)))],axis=1);return [float(x) for x in np.quantile(v,[.025,.975])]
def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-w,c+w]
def summarize_a(out):
    labels={r['audit_id']:r for r in csv.DictReader((out/'audit/AI_LABELS.csv').open(encoding='utf-8-sig'))};pairs=read(out/'AUDIT_SELECTION_PRIVATE.json')['pairs'];sim={r['audit_id']:r for r in read(out/'SIMULATOR_RECONSTRUCTION.json')['records']};strata=[]
    assert set(labels)==set(sim)
    for r in pairs:assert labels[r['audit_id']]['image_sha256']==sim[r['audit_id']]['image_sha256'] and labels[r['audit_id']]['reviewer_type']=='AI'
    for task in [7,8,9]:
        for cl in ['SHARE','BIF']:
            for gate in [True,False]:
                selected=[r for r in pairs if r['task']==task and r['label']==cl and r['physical_pass']==gate];k=sum(labels[r['audit_id']]['label']=='COMPARABLE' for r in selected);u=sum(labels[r['audit_id']]['label']=='UNCERTAIN' for r in selected)
                strata.append(dict(task=task,action_label=cl,physical_pass=gate,n=len(selected),comparable=k,uncertain=u,fraction=k/len(selected) if selected else None,wilson95=wilson(k,len(selected))))
    cls=[]
    for cl in ['SHARE','BIF']:
        rr=[r for r in strata if r['action_label']==cl and r['physical_pass']];n=sum(r['n'] for r in rr);k=sum(r['comparable'] for r in rr);ci=wilson(k,n);cls.append(dict(label=cl,n=n,comparable=k,fraction=k/n,descriptive_wilson95=ci,gate=k/n>=.8 and ci[0]>=.6))
    report=dict(status='AI_EXPLORATORY_SUPPORT' if all(r['gate'] for r in cls) else 'LABEL_RELIABILITY_NOT_ESTABLISHED',reviewer='AI_NOT_HUMAN',n=len(labels),counts=dict(collections.Counter(r['label'] for r in labels.values())),by_stratum=strata,by_class=cls,uncertainty='Fixed diversity stratified audit; Wilson intervals are descriptive, not independent-population coverage; no human ground truth.',multi_frame_contact_evidence=True)
    write(out/'R5_A_REPORT.json',report);return labels,report
def summarize_b(out,labels):
    rows=[read(p)['row'] for p in sorted((out/'B').glob('*/*.json'))];table(out/'R5_B_RAW.csv',rows);reports=[];allpred=[];allmetrics=[]
    for subset in ['physical_pass','AI_comparable_physical','all_resolved_diagnostic']:
        data=[r for r in rows if subset=='all_resolved_diagnostic' or (r['physical_pass'] and (subset=='physical_pass' or labels[r['audit_id']]['label']=='COMPARABLE'))]
        preds=[];skipped=0
        for r in data:
            tr=[s for s in data if no_leak(s,r)]
            if len(tr)<20 or len(set(s['start_id'] for s in tr))<4:skipped+=1;continue
            p=dict(r,subset=subset,train_rows=len(tr),train_checkpoints=len(set(s['start_id'] for s in tr)))
            for name,features in PROTOCOL['B']['models'].items():p['pred_'+name]=predict(tr,r,features)
            preds.append(p)
        metrics=[]
        for cp in sorted(set(r['start_id'] for r in preds)):
            rr=[r for r in preds if r['start_id']==cp];y=np.array([r['old_delta'] for r in rr]);db=np.array([r['deadband'] for r in rr]);m=dict(subset=subset,start_id=cp,seed=rr[0]['seed'],task=rr[0]['task'],n=len(rr))
            for name in PROTOCOL['B']['models']:
                p=np.array([r['pred_'+name] for r in rr]);valid=(abs(p)>db)&(abs(y)>db);m[name+'_spearman']=corr(p,y);m[name+'_mae']=float(np.mean(abs(p-y)));m[name+'_sign']=float(np.mean(np.sign(p[valid])==np.sign(y[valid]))) if valid.any() else None
            m['action_added_spearman']=(m['physical_plus_action_spearman']-m['physical_spearman']) if m['physical_plus_action_spearman'] is not None and m['physical_spearman'] is not None else None
            m['raw_hdot_sign']=float(np.mean([np.sign(r['h_dot_d'])==np.sign(r['old_delta']) for r in rr if abs(r['h_dot_d'])>r['deadband'] and abs(r['old_delta'])>r['deadband']])) if any(abs(r['h_dot_d'])>r['deadband'] and abs(r['old_delta'])>r['deadband'] for r in rr) else None
            metrics.append(m)
        dif=[m['action_added_spearman'] for m in metrics if m['action_added_spearman'] is not None];ci=boot(dif);cells=[]
        for seed in [45,46,47]:
            for task in [7,8]:
                dd=[m['action_added_spearman'] for m in metrics if m['seed']==seed and m['task']==task and m['action_added_spearman'] is not None];cells.append(dict(seed=seed,task=task,delta_mean=float(np.mean(dd)) if dd else None,n=len(dd)))
        # Sensitivity cluster: a pair of qualified trajectories, independent of checkpoint repetitions.
        groups=collections.defaultdict(list)
        for r in preds:groups[r['new_trajectory']+'|'+r['old_trajectory']].append(abs(r['pred_physical']-r['old_delta'])-abs(r['pred_physical_plus_action']-r['old_delta']))
        tg=[float(np.mean(v)) for v in groups.values()]
        gate=len(dif)>=8 and float(np.median(dif))>=.1 and ci is not None and ci[0]>0 and sum(c['delta_mean'] is not None and c['delta_mean']>0 for c in cells)>=5
        reports.append(dict(subset=subset,n_measurements=len(data),n_unique_pairs=len(set(r['pair_id'] for r in data)),n_checkpoints=len(set(r['start_id'] for r in data)),n_crossfit=len(preds),n_no_legal_training_fold=skipped,action_incremental_status='SUPPORTED' if gate else ('INCONCLUSIVE_INSUFFICIENT_GROUPED_DATA' if len(dif)<8 else 'NOT_ESTABLISHED'),action_added_spearman_median=float(np.median(dif)) if dif else None,checkpoint_mean_delta95=ci,trajectory_pair_MAE_improvement_mean95=boot(tg),seed_task_cells=cells,predictors={name:dict(median_spearman=float(np.median([m[name+'_spearman'] for m in metrics if m[name+'_spearman'] is not None])) if any(m[name+'_spearman'] is not None for m in metrics) else None) for name in PROTOCOL['B']['models']}))
        allpred.extend(preds);allmetrics.extend(metrics)
    table(out/'R5_B_CROSSFIT.csv',allpred);table(out/'R5_B_CHECKPOINT_METRICS.csv',allmetrics);report=dict(primary=reports[0],secondary=reports[1:],no_cross_validation_leakage='Training excludes BOTH the test checkpoint and EITHER task-qualified trajectory.',target='Actual old NLL change, not SHARE/BIF agreement',task9='NOT_RUN_NO_COMPLETE_OPTIMIZER_STATE',formal_training_resumed=False);write(out/'R5_B_REPORT.json',report);return report
def summarize_c(out):
    rows=[read(p) for p in sorted((out/'C').glob('*/*.json')) if p.name!='NOT_RUN.json'];table_rows=[];wins=[]
    native={r['start_id']:r for r in rows if r['arm']=='MLR'}
    for r in rows:
        n=native[r['start_id']];base={(v['kind'],v['origin']):v for v in n['observations']};old=[v for v in r['observations'] if v['kind']=='old'];new=[v for v in r['observations'] if v['kind']=='new']
        olddiff=[max(v['delta'],0)-max(base[v['kind'],v['origin']]['delta'],0) for v in old];newdiff=[v['delta']-base[v['kind'],v['origin']]['delta'] for v in new]
        row=dict(start_id=r['start_id'],seed=r['seed'],task=r['task'],arm=r['arm'],share_pairs=r['pair_counts']['SHARE'],bif_pairs=r['pair_counts']['BIF'],native_equal=r['bitwise_equal_native'],old_probe_n=len(old),new_probe_n=len(new),old_positive_damage=float(np.mean([max(v['delta'],0) for v in old])) if old else None,old_damage_difference=float(np.mean(olddiff)) if old else None,new_delta=float(np.mean([v['delta'] for v in new])) if new else None,plasticity_cost=float(np.mean(newdiff)) if new else None,seconds=r['seconds']);table_rows.append(row)
        for v in r['observations']:wins.append(dict(start_id=r['start_id'],seed=r['seed'],arm=r['arm'],**v))
    summaries=[]
    for arm in ['SHARE_ONLY','BIF_ONLY','ABCR']:
        rr=[r for r in table_rows if r['arm']==arm];active=sum((r['share_pairs'] if arm=='SHARE_ONLY' else r['bif_pairs'] if arm=='BIF_ONLY' else r['share_pairs']+r['bif_pairs'])>0 for r in rr);dd=[r['old_damage_difference'] for r in rr if r['old_damage_difference'] is not None];ci=boot(dd)
        summaries.append(dict(arm=arm,n=len(rr),active_checkpoints=active,bitwise_equal_native=sum(r['native_equal'] for r in rr),old_damage_difference_median=float(np.median(dd)) if dd else None,checkpoint_bootstrap95=ci,status='INCONCLUSIVE_INSUFFICIENT_ACTIVATION' if active<8 else 'REQUIRES_FULL_PREREGISTERED_EFFICACY_CHECK'))
    table(out/'R5_C_CHECKPOINT_METRICS.csv',table_rows);table(out/'R5_C_PROBE_RESULTS.csv',wins);report=dict(arms=summaries,branches=len(rows),checkpoints=len(native),native_parity_all=all(all(v for v in r['native_parity'].values()) for r in rows if r['native_parity']),restore_all=all(all(r['restored'].values()) for r in rows),definition='Investigator-defined exploratory loss; frozen before outcomes; unmodified exact mixed batches; old side detached; no rebalancing or retrieval',formal_training_resumed=False);write(out/'R5_C_REPORT.json',report);return report
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();labels,A=summarize_a(a.out);B=summarize_b(a.out,labels);C=summarize_c(a.out);write(a.out/'SUMMARY.json',dict(status='COMPLETED_WITH_LIMITATIONS',R5_A=A['status'],R5_B=B['primary']['action_incremental_status'],R5_C={r['arm']:r['status'] for r in C['arms']},ABCR_FULL_TRAINING_ALLOWED=False,formal_training_resumed=False))

if __name__=='__main__':main()
