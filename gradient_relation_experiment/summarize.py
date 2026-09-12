"""Isolated loss labels, descriptive prediction metrics and leave-one-seed-out tests."""
import itertools
import math
import time
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from cb_ifa_experiment.gpu_runtime import ROOT,read,write,sha
from gradient_relation_experiment.run import verify_code

OUT=ROOT/'reports/gradient_relation_v1'
TARGETS=('E_new_to_old','E_old_to_new','H_MLR')


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def metrics(y,score,tolerance):
    y=np.asarray(y,dtype=float);score=np.asarray(score,dtype=float);tolerance=np.asarray(tolerance,dtype=float)
    ok=np.isfinite(y)&np.isfinite(score)&np.isfinite(tolerance);y=y[ok];score=score[ok];tolerance=tolerance[ok]
    if not len(y):return {'n':0}
    harm=y>tolerance;pred=score>0;both=len(set(harm))==2
    rank=float(spearmanr(y,score).statistic) if len(set(y))>1 and len(set(score))>1 else None
    return clean({'n':len(y),'harm_count':int(harm.sum()),'help_count':int((y<-tolerance).sum()),
        'neutral_count':int((abs(y)<=tolerance).sum()),'harm_prevalence':float(harm.mean()),
        'spearman':rank,'ROC_AUC':float(roc_auc_score(harm,score)) if both else None,
        'PR_average_precision':float(average_precision_score(harm,score)) if both else None,
        'zero_threshold_precision':float(precision_score(harm,pred,zero_division=0)) if pred.any() else None,
        'zero_threshold_recall':float(recall_score(harm,pred,zero_division=0)) if harm.any() else None,
        'predicted_harm_count':int(pred.sum()),'mean_loss_difference':float(y.mean()),'median_loss_difference':float(np.median(y))})


def read_rows(name):
    rows=[]
    for seed in (45,46,47):
        p=ROOT/f'runs/gradient_relation_v1/seed{seed}/{name}.jsonl'
        with p.open() as f:rows.extend(__import__('json').loads(line) for line in f if line.strip())
    return rows


def save_csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False)


def run():
    verify_code();runs=[]
    for seed in (45,46,47):
        d=ROOT/f'runs/gradient_relation_v1/seed{seed}';r=read(d/'COMPLETE.json')
        assert r['status']=='PASS' and r['points_completed']==40 and r['carrier_source']=='new_controlled_3090_from_official'
        for name,h in r['files'].items():assert sha(d/name)==h
        runs.append(r)
    assert len({r['initial_model_sha256'] for r in runs})==1
    scores=pd.DataFrame(read_rows('scores'));labels=pd.DataFrame(read_rows('labels'))
    checks=read_rows('checks');carrier_checks=read_rows('carrier_checks')
    assert len(checks)==120 and len({r['point'] for r in checks})==120
    assert all(r['carrier_restored_exact'] and r['native_11_parity'] for r in checks)
    assert all(all(r['matches'].values()) for r in carrier_checks)
    df=labels.merge(scores,on=['point','seed','task','phase','old_task'],validate='many_to_one')
    df['phase_late']=(df.phase=='70-99').astype(int)
    df['negative_cosine']=-df.cosine;df['negative_dot']=-df.new_dot_old;df['negative_mixed_dot']=-df.mixed_dot_old
    summaries=[];raw_metrics=[]
    grouping=[([],[((),df)])]
    for cols in (['seed'],['task'],['phase'],['seed','task','phase']):grouping.append((cols,df.groupby(cols,sort=True)))
    for cols,groups in grouping:
        for key,frame in groups:
            if not isinstance(key,tuple):key=(key,)
            group=dict(zip(cols,key));group['grouping']='+'.join(cols) or 'all'
            for (bank,horizon),g in frame.groupby(['bank','horizon']):
                for target in TARGETS:
                    tol=g['new_tolerance' if target=='E_old_to_new' else 'old_tolerance']
                    predictors=['negative_cosine','negative_dot','negative_mixed_dot','old_dot_d11']
                    predictors+=['new_dot_d11_minus_d10' if target=='E_old_to_new' else 'old_dot_d11_minus_d01']
                    for score in predictors:
                        raw_metrics.append(dict(group,bank=bank,horizon=int(horizon),target=target,predictor=score,**metrics(g[target],g[score],tol)))
                summaries.append(dict(group,bank=bank,horizon=int(horizon),n=len(g),
                    mean_E_new_to_old=float(g.E_new_to_old.mean()),mean_E_old_to_new=float(g.E_old_to_new.mean()),mean_H_MLR=float(g.H_MLR.mean()),
                    bidirectional_help_count=int(((g.E_new_to_old < -g.old_tolerance)&(g.E_old_to_new < -g.new_tolerance)).sum()),
                    interference_but_MLR_not_damaged_count=int(((g.E_new_to_old>g.old_tolerance)&(g.H_MLR<=g.old_tolerance)).sum())))
    save_csv('label_distribution.csv',summaries);save_csv('raw_prediction_metrics.csv',raw_metrics)
    df.to_csv(OUT/'joined_observations.csv',index=False)
    # Basic information excludes seed; each test seed is held out as a whole.
    base=['task','old_task','phase_late','epoch','lr','n_new','n_old_j','n_batch']
    fits=[];predictions=[]
    for (bank,horizon),g0 in df.groupby(['bank','horizon']):
        g=g0[g0.gradient_valid.astype(bool)].copy()
        for target in TARGETS:
            disp='new_dot_d11_minus_d10' if target=='E_old_to_new' else 'old_dot_d11_minus_d01' if target=='E_new_to_old' else 'old_dot_d11'
            definitions={'base':base,'base_mixed':base+['mixed_dot_old'],
                'base_mixed_cos':base+['mixed_dot_old','cosine'],
                'base_mixed_cos_amplitude':base+['mixed_dot_old','cosine','new_dot_old','new_norm','old_norm'],
                'base_mixed_displacement':base+['mixed_dot_old',disp],
                'base_mixed_displacement_cos':base+['mixed_dot_old',disp,'cosine']}
            tolerance='new_tolerance' if target=='E_old_to_new' else 'old_tolerance'
            for held in (45,46,47):
                tr=g[g.seed!=held];te=g[g.seed==held]
                for name,features in definitions.items():
                    if not len(tr) or not len(te):continue
                    assert np.isfinite(tr[features].values).all() and np.isfinite(te[features].values).all()
                    reg=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                    reg.fit(tr[features],tr[target]);pred=reg.predict(te[features])
                    observed=(te[target].values>te[tolerance].values).astype(int)
                    ytrain=(tr[target].values>tr[tolerance].values).astype(int)
                    logits=None
                    if len(set(ytrain))==2:
                        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=20260913))
                        clf.fit(tr[features],ytrain);logits=clf.decision_function(te[features])
                        assert int(clf[-1].n_iter_[0])<2000,'Classifier did not converge'
                    record={'bank':bank,'horizon':int(horizon),'target':target,'held_out_seed':held,'model':name,
                        'train_n':len(tr),'test_n':len(te),'MSE':float(mean_squared_error(te[target],pred)),
                        'MAE':float(mean_absolute_error(te[target],pred)),
                        'ROC_AUC':float(roc_auc_score(observed,logits)) if logits is not None and len(set(observed))==2 else None,
                        'PR_average_precision':float(average_precision_score(observed,logits)) if logits is not None and len(set(observed))==2 else None,
                        'classifier_fitted':logits is not None,'features':';'.join(features)}
                    fits.append(record)
                    for i,(_,row) in enumerate(te.iterrows()):predictions.append({'point':row.point,'old_task':row.old_task,'bank':bank,
                        'task':row.task,'phase':row.phase,
                        'horizon':int(horizon),'target':target,'held_out_seed':held,'model':name,'observed':float(row[target]),
                        'harm':int(observed[i]),'predicted_difference':float(pred[i]),'classification_logit':float(logits[i]) if logits is not None else None})
    save_csv('LOSO_metrics.csv',fits);save_csv('LOSO_predictions.csv',predictions)
    subfits=[];prediction_frame=pd.DataFrame(predictions)
    for grouping in ('phase','task'):
        keys=['bank','horizon','target','held_out_seed','model',grouping]
        for key,g in prediction_frame.groupby(keys):
            row=dict(zip(keys,key));row.update(grouping=grouping,group=str(row.pop(grouping)),n=len(g),
                MSE=float(mean_squared_error(g.observed,g.predicted_difference)),
                MAE=float(mean_absolute_error(g.observed,g.predicted_difference)),
                PR_average_precision=float(average_precision_score(g.harm,g.classification_logit)) if g.classification_logit.notna().all() and g.harm.nunique()==2 else None)
            subfits.append(row)
    save_csv('LOSO_subgroups.csv',subfits)
    increments=[];fit=pd.DataFrame(fits)
    for basic,extended in [('base','base_mixed'),('base_mixed','base_mixed_cos'),('base_mixed_displacement','base_mixed_displacement_cos')]:
        keys=['bank','horizon','target','held_out_seed']
        z=fit[fit.model==basic].merge(fit[fit.model==extended],on=keys,suffixes=('_base','_extended'),validate='one_to_one')
        for _,row in z.iterrows():
            increments.append({**{k:row[k] for k in keys},'comparison':extended+' minus '+basic,
                'MSE_reduction':row.MSE_base-row.MSE_extended,'MAE_reduction':row.MAE_base-row.MAE_extended,
                'ROC_AUC_gain':row.ROC_AUC_extended-row.ROC_AUC_base,
                'PR_average_precision_gain':row.PR_average_precision_extended-row.PR_average_precision_base})
    save_csv('incremental_value.csv',increments)
    stability=[]
    for _,r in scores.iterrows():
        valid=[x for x in r.stability_cosines if x is not None and math.isfinite(x)]
        flip=[(x<0)!=(r.cosine<0) for x in valid] if r.cosine is not None and math.isfinite(r.cosine) else []
        stability.append({'point':r.point,'seed':r.seed,'task':r.task,'phase':r.phase,'old_task':r.old_task,
            'valid_resamples':len(flip),'sign_flip_count':sum(flip),'sign_flip_rate':sum(flip)/len(flip) if flip else None,
            'original_cosine':r.cosine,'resampled_cosines':';'.join(str(x) for x in valid)})
    save_csv('reference_stability.csv',stability)
    inc=pd.DataFrame(increments)
    def consistent(target,comparison):
        z=inc[(inc.horizon==20)&(inc.target==target)&(inc.comparison==comparison)]
        details={};ok=True
        for bank in ('A','B'):
            g=z[z.bank==bank]
            good=len(g)==3 and g.MSE_reduction.mean()>0 and (g.MSE_reduction>0).sum()>=2
            cls=g.PR_average_precision_gain.notna().all() and g.PR_average_precision_gain.mean()>0 and (g.PR_average_precision_gain>0).sum()>=2
            ok=bool(ok and good and cls)
            details[bank]={'mean_MSE_reduction':g.MSE_reduction.mean(),'MSE_improved_seeds':int((g.MSE_reduction>0).sum()),
                'mean_PR_gain':g.PR_average_precision_gain.mean(),'PR_improved_seeds':int((g.PR_average_precision_gain>0).sum())}
        extended,basic=comparison.split(' minus ')
        sg=pd.DataFrame(subfits);sg=sg[(sg.horizon==20)&(sg.target==target)]
        subkeys=['bank','horizon','target','held_out_seed','grouping','group']
        pairs=sg[sg.model==basic].merge(sg[sg.model==extended],on=subkeys,suffixes=('_base','_extended'))
        subgroup_details=[]
        for key,g in pairs.groupby(['bank','grouping','group']):
            improvement=float((g.MSE_base-g.MSE_extended).mean())
            ok=bool(ok and len(g)==3 and improvement>0)
            subgroup_details.append({'bank':key[0],'grouping':key[1],'group':key[2],'mean_MSE_reduction':improvement})
        details['task_phase_checks']=subgroup_details
        return ok,clean(details)
    evidence={}
    for target in TARGETS:
        mix,md=consistent(target,'base_mixed_cos minus base_mixed')
        opt,od=consistent(target,'base_mixed_displacement_cos minus base_mixed_displacement')
        evidence[target]={'descriptively_consistent_beyond_mixed':mix,'descriptively_consistent_beyond_displacement':opt,
            'beyond_mixed':md,'beyond_displacement':od}
    h=evidence['H_MLR']
    if h['descriptively_consistent_beyond_mixed'] and h['descriptively_consistent_beyond_displacement']:
        route='DESCRIPTIVE_INCREMENTAL_SIGNAL_CANDIDATE_REQUIRES_CLOSED_LOOP'
    elif any(evidence[t]['descriptively_consistent_beyond_mixed'] for t in TARGETS[:2]):
        route='LOCAL_RELATION_SIGNAL_ONLY_PROTECTION_VALUE_NOT_ESTABLISHED'
    else:route='INCREMENTAL_PREDICTION_VALUE_NOT_ESTABLISHED'
    decision={'route':route,'evidence':evidence,'carrier':'new controlled3090; not historical4090',
        'classification_rule':'descriptive candidate requires positive mean and >=2/3 held-out seeds improved MSE and PR on both banks at20, plus positive mean MSE increment within each task and phase; not statistical significance',
        'closed_loop_supported':False,'controller_launch':False,'rollout_launch':False,
        'caveat':'Three correlated-seed trajectories; 120 origins are not120 independent replications. Failure of this model does not prove absence of every nonlinear signal.'}
    write(OUT/'DECISION.json',clean(decision))
    lines=['# 梯度关系与同状态反事实：机制复核报告','',
        '120个预选目标起点已完成，三个新生成的3090原生MLR载体各40点；不是历史4090训练状态。',
        '四分支仅首步贡献不同，其后19步全部原MLR；第20步为主终点，A/B在完整20步训练及参考轨迹之外。',
        '真实载体与无干预11分支的第1/5/20步模型、moments、scheduler和RNG核对通过。','',
        '## 描述性判定','',route,'','没有闭环rollout，因此没有建立机器人能力层面的共享/负迁移结论，也不启动控制器。','',
        '## 第20步方向性影响','',
        '| bank | seed | E 新→旧均值 | E 旧→新均值 | H 原MLR旧损伤均值 | 样本对 |','|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        if row['grouping']=='seed' and row['horizon']==20:
            lines.append(f"| {row['bank']} | {row['seed']} | {row['mean_E_new_to_old']:.7g} | {row['mean_E_old_to_new']:.7g} | {row['mean_H_MLR']:.7g} | {row['n']} |")
    lines+=['','正值表示对应的独立损失增加；三个量是不同标签。负NLL的差值照常计算，不用原损失作百分比分母。','',
        '## 增量预测检查','',__import__('json').dumps(clean(evidence),ensure_ascii=False,indent=2),'',
        '完整分seed、task、阶段结果见raw_prediction_metrics.csv和label_distribution.csv。',
        'LOSO_metrics.csv及incremental_value.csv比较固定预测器，全部seed按整体留出，没有调参或按窗口随机切分。',
        'reference_stability.csv记录额外三组buffer参考的符号翻转；equal_norm.jsonl是仅单步的人工幅度对照。','',
        '## 限制与解释','',
        '本轮是同状态有限位移机制干预，包含AdamW历史矩、clip和weight decay，不能将差异归因于纯梯度方向。',
        '无共同当前/旧任务或参考范数无效时，不把缺失记成零冲突；相关观察覆盖数见PREPARED.json。',
        '三个seed是已有开发seed；A/B只在这20步内未访问，不是历史从未见数据。',
        '同一task/phase和同seed起点相关，任务对观察数量不等于独立统计样本量。',
        '重复测量/无干预差及浮点基准定义容差；容差不是显著性检验。',
        'ROC-AUC指损伤分类预测，PR值采用average_precision_score，不是闭环成功率AUC。',
        '当前3090重放历史4090失败已保留，用户随后授权新受控载体。不得将新载体结果写回原历史实验。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'COMPLETE.json',{'status':'PASS','time':time.time(),'target_points':120,'completed_points':120,
        'observed_task_pairs':len(scores),'banks':2,'horizons':[1,5,20],'native_carrier_steps':sum(r['native_steps'] for r in runs),
        'all_live_carrier_checks_passed':True,'decision':route,'scientific_scope':'local mechanism only; no controller/no rollout'})
    with zipfile.ZipFile(OUT/'FINAL_DELIVERY.zip','w',zipfile.ZIP_DEFLATED) as z:
        for base in [OUT,ROOT/'gradient_relation_experiment',ROOT/'runs/gradient_relation_v1']:
            for p in base.rglob('*'):
                if p.is_file() and p.suffix in ('.py','.md','.json','.csv','.jsonl','.gz','.txt') and '__pycache__' not in p.parts and p.name!='QUEUE_STATUS.json':z.write(p,str(p.relative_to(ROOT)))


if __name__=='__main__':run()
