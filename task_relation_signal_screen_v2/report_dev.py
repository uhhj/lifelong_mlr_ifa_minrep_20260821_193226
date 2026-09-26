"""Development-only calibration, ranking, point decisions and transparent plots."""
import csv
import itertools
import numpy as np
from scipy.stats import spearmanr,kendalltau
from .common import *

def records(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def rank_metrics(observed,predicted):
    a=np.asarray(observed);b=np.asarray(predicted)
    pairs=[(i,j) for i,j in itertools.combinations(range(len(a)),2) if a[i]!=a[j]]
    return dict(informative=bool(pairs),pair_count=len(pairs),
        spearman=float(spearmanr(a,b).statistic) if pairs and np.ptp(b)>0 else None,
        kendall=float(kendalltau(a,b).statistic) if pairs and np.ptp(b)>0 else None,
        pairwise_accuracy=float(np.mean([.5 if b[i]==b[j] else ((a[i]-a[j])*(b[i]-b[j])>0) for i,j in pairs])) if pairs else None)

def discrete_effect(values,trials=20):
    values=np.asarray(values);counts=np.rint(values*trials)
    assert np.max(abs(values-counts/trials),initial=0)<1e-12
    # Exactly one of twenty trials is exactly5pp, not slightly >5pp by subtraction rounding.
    return np.where(counts>.05*trials,1,np.where(counts<-.05*trials,-1,0))

def label_uncertainty():
    from .confirmation_statistics import reconstruct
    data=records(OLD/'ROLLOUTS.csv');success={}
    for row in data:success.setdefault((int(row['seed']),row['model_id'],int(row['task'])),[]).append((int(row['trial']),int(row['success'])))
    success={key:np.array([v for i,v in sorted(vals)]) for key,vals in success.items()}
    assert all(len(a)==20 for a in success.values())
    rr=[dict(seed=k,source=s,target=t) for k,s,t in PAIRS];rng=np.random.RandomState(2026092603);values=[]
    for _ in range(1000):
        indices={(k,t):rng.randint(0,20,20) for k in SEEDS for t in TASKS}
        values.append(reconstruct(rr,success,indices))
    values=np.asarray(values);lo,hi=np.quantile(values,[.025,.975],axis=0)
    result=[]
    for i,row in enumerate(rr):
        for j,e in enumerate(('TRANSFER_300','DAMAGE_300')):
            a,b=round(float(lo[i,j]),12),round(float(hi[i,j]),12)
            label='POSITIVE' if a>.05 else 'NEGATIVE' if b<-.05 else 'WITHIN_BAND' if a>=-.05 and b<=.05 else 'UNCERTAIN'
            result.append(dict(row,endpoint=e,lower=a,upper=b,label=label,interval='pointwise95% paired-trial bootstrap,1000 fixed repeats, old20-trial panel; not simultaneous candidate confirmation'))
    table(OUT/'DEV_LABEL_UNCERTAINTY.csv',result)
    return {(r['seed'],r['source'],r['target'],r['endpoint']):r for r in result}

def main():
    rows=records(OUT/'DEV_OOF_PREDICTIONS.csv')
    for r in rows:
        for key in ('seed','source','target'):r[key]=int(r[key])
        for key in ('observed','predicted','predicted_unclipped'):r[key]=float(r[key])
    uncertainty=label_uncertainty()
    calibration=[];rankings=[];risk=[];regrets=[];coverage=[]
    for endpoint in ('TRANSFER_300','DAMAGE_300'):
        names=sorted(set(r['model'] for r in rows if r['outcome']==endpoint))
        for name in names:
            rr=[r for r in rows if r['outcome']==endpoint and r['model']==name and r['mode']=='task']
            if len(rr)!=36:continue
            y=np.array([r['observed'] for r in rr]);p=np.array([r['predicted'] for r in rr]);raw=np.array([r['predicted_unclipped'] for r in rr])
            slope=float(np.cov(p,y,bias=True)[0,1]/np.var(p)) if np.ptp(p)>0 else None
            calibration.append(dict(endpoint=endpoint,model=name,mse=float(np.mean((p-y)**2)),unclipped_mse=float(np.mean((raw-y)**2)),mae=float(np.mean(abs(p-y))),slope=slope,intercept=float(y.mean()-slope*p.mean()) if slope is not None else None,saturation_count=int(np.sum(abs(raw)>1)),scope='development selected panel'))
            pred_risk=p<-.05 if endpoint=='TRANSFER_300' else p>.05
            measured_class=discrete_effect(y)
            true_risk=measured_class<0 if endpoint=='TRANSFER_300' else measured_class>0
            tp=int(np.sum(pred_risk & true_risk));fp=int(np.sum(pred_risk & ~true_risk));fn=int(np.sum(~pred_risk & true_risk))
            risk.append(dict(endpoint=endpoint,model=name,units=36,point_predicted_risk=int(pred_risk.sum()),point_true_risk=int(true_risk.sum()),true_positive=tp,false_positive=fp,false_negative=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,point_uncertain_band_fraction=float(np.mean(abs(p)<=.05)),false_suppression_positive_transfer=int(np.sum((p<-.05)&(measured_class>0))) if endpoint=='TRANSFER_300' else None,warning='point rule, not a calibrated safety probability; 36 rows are only 12 directed relations'))
            identifiable=[uncertainty[r['seed'],r['source'],r['target'],endpoint]['label']!='UNCERTAIN' for r in rr]
            coverage.append(dict(endpoint=endpoint,model=name,total=36,identifiable=sum(identifiable),coverage=sum(identifiable)/36,uncertain=36-sum(identifiable),continuous_metrics_include_all=True))
            for fixed in (('target','source') if endpoint=='DAMAGE_300' else ('target',)):
                for seed in SEEDS:
                    for task in TASKS:
                        group=[r for r in rr if r['seed']==seed and r[fixed]==task]
                        rankings.append(dict(endpoint=endpoint,model=name,seed=seed,fixed=fixed,task=task,**rank_metrics([r['observed'] for r in group],[r['predicted'] for r in group])))
            if endpoint=='TRANSFER_300':
                for seed in SEEDS:
                    for target in TASKS:
                        group=[r for r in rr if r['seed']==seed and r['target']==target]
                        best=min(group,key=lambda r:(-r['predicted'],r['source']))
                        choose=best['predicted']>.05;gain=best['observed'] if choose else 0.;oracle=max([0.]+[r['observed'] for r in group])
                        regrets.append(dict(model=name,seed=seed,target=target,chosen_source=best['source'] if choose else 'BASE',selected_transfer=gain,regret=oracle-gain,no_source_regret=oracle,uniform_source_expected_regret=oracle-float(np.mean([r['observed'] for r in group])),rule='max predicted transfer if >5pp else base; truth only evaluates'))
    table(OUT/'DEV_CALIBRATION.csv',calibration);table(OUT/'DEV_RANKING.csv',rankings)
    table(OUT/'DEV_RISK_CLASSIFICATION.csv',risk);table(OUT/'DEV_DECISION_REGRET.csv',regrets)
    table(OUT/'DEV_UNCERTAIN_COVERAGE.csv',coverage)
    # Signal repeatability is descriptive, not independent replications of task pairs.
    rel=records(OUT/'SIGNAL_RELIABILITY.csv');repeatability=[]
    for field in sorted(set(r['feature'] for r in rel)):
        rr=[r for r in rel if r['feature']==field and r['D1'] and r['D2']]
        a=np.array([float(r['D1']) for r in rr]);b=np.array([float(r['D2']) for r in rr])
        repeatability.append(dict(feature=field,available_pairs=len(rr),missing_pairs=36-len(rr),pearson=float(np.corrcoef(a,b)[0,1]) if len(rr)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None,sign_agreement=float(np.mean(np.sign(a)==np.sign(b))) if len(rr) else None,mean_abs_half_difference=float(np.mean(abs(a-b))) if len(rr) else None,half_definition='D1/D2 measurements; readout uses reciprocal cross-fit held halves'))
    table(OUT/'SIGNAL_REPEATABILITY_SUMMARY.csv',repeatability)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir=OUT/'figures';figure_dir.mkdir(exist_ok=True)
    for endpoint in ('TRANSFER_300','DAMAGE_300'):
        fig,axes=plt.subplots(3,5,figsize=(18,11),layout='constrained')
        for ax,name in zip(axes.flat,list(PACKAGES)+['B0']):
            rr=[r for r in rows if r['outcome']==endpoint and r['model']==name and r['mode']=='task']
            matrix=np.full((4,4),np.nan)
            for si,s in enumerate(TASKS):
                for ti,t in enumerate(TASKS):
                    vals=[r['predicted'] for r in rr if r['source']==s and r['target']==t]
                    if vals:matrix[si,ti]=np.mean(vals)
            ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_title(name+' OOF prediction')
            ax.set_xticks(range(4),TASKS);ax.set_yticks(range(4),TASKS);ax.set_xlabel('target');ax.set_ylabel('source')
            for si,ti in itertools.product(range(4),repeat=2):
                if np.isfinite(matrix[si,ti]):ax.text(ti,si,'%.2f'%matrix[si,ti],ha='center',va='center',fontsize=9)
        fig.suptitle(endpoint+' | DEVELOPMENT only | mean across 3 optimization seeds')
        fig.savefig(figure_dir/(endpoint+'_predictions.png'),dpi=150);plt.close(fig)
    raw=records(OUT/'SIGNALS_RAW.csv');fields=[f for values in PACKAGES.values() for f in values]
    fig,axes=plt.subplots(5,6,figsize=(21,17),layout='constrained')
    for ax,field in zip(axes.flat,fields):
        matrix=np.full((4,4),np.nan)
        for si,s in enumerate(TASKS):
            for ti,t in enumerate(TASKS):
                vals=[float(r[field]) for r in raw if int(r['source'])==s and int(r['target'])==t and r.get(field)]
                if vals:matrix[si,ti]=np.mean(vals)
        img=ax.imshow(matrix,cmap='viridis');ax.set_title(field,fontsize=9);ax.set_xticks(range(4),TASKS);ax.set_yticks(range(4),TASKS);fig.colorbar(img,ax=ax,shrink=.6)
    for ax in list(axes.flat)[len(fields):]:ax.axis('off')
    fig.suptitle('Raw signals, separate scale per panel; rows source / columns target; not predictions')
    fig.savefig(figure_dir/'RAW_SIGNAL_MATRICES.png',dpi=130);plt.close(fig)
    finalists=read(OUT/'FINALISTS_FROZEN.json');stage=read(OUT/'DEVELOPMENT_SUMMARY.json')
    text=['# 关系信号系统筛选 v2：Stage A 开发阶段报告','',
          '本文件报告旧4任务、12种有向关系、3个优化seed的开发筛选。独立6任务确认尚未执行，不能据此宣布预测价值已确认。',
          '', '原2880次闭环结果只读复用；本轮新产生的是14包信号与原训练轨迹的短前缀。有效更新与原调用步分开记录。',
          '', '## 冻结候选', '', '|终点|候选角色|包|开发MSE|','|---|---|---|---:|']
    text += ['|%s|%s|%s|%.6g|'%(r['endpoint'],r['role'],r['package'],r['development_mse']) for r in finalists['candidates']]
    text += ['', '所有候选、原始值、交叉验证预测、缺失性、半集重复性、排序、风险误判与选择遗憾均分别落盘。主变换仅使用训练折；动态组合的外层验证内重新进行C0选择。',
             '', 'COMPUTE_COST.csv 为共享B0前向之后的增量计算；S10计入前FiLM缓存成本（按冻结36行中各任务18次出现分摊），缓存原始耗时另存。不据该增量时间宣称部署总成本优势。',
             '', '1000次无序任务对块置换与seed/无序任务对bootstrap是开发描述性检查，不能替代独立面板的5000次配对同时区间。',
             '', '## 下一阶段与证据边界','',
             'GOAL/SPATIAL面板必须先完成任务与数据身份冻结、预算和最终预测器封存，再运行新任务训练及9360次闭环。Stage A选择不依赖显著性门；没有独立确认结果时，所有新任务价值裁决均为NOT_ESTABLISHED。',
             '', 'formal_training_resumed=false','diagnostic_training_performed=true']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(OUT/'SUMMARY.json',dict(stage='STAGE_A_COMPLETE',study_complete=False,development=stage,confirmation='NOT_RUN',finalists=len(finalists['candidates']),CALIBRATED_FORECAST_VALUE='NOT_ESTABLISHED',SOURCE_RANKING_VALUE='NOT_ESTABLISHED',NEGATIVE_TRANSFER_WARNING_VALUE='NOT_ESTABLISHED',OLD_DAMAGE_WARNING_VALUE='NOT_ESTABLISHED',PROBE_INCREMENT_VALUE='NOT_ESTABLISHED',COST_VALUE='NOT_ESTABLISHED',formal_training_resumed=False,diagnostic_training_performed=True))

if __name__=='__main__':main()
