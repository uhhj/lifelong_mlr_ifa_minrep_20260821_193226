"""Frozen-panel statistics. Never fits predictors or reads labels in signal jobs."""
import itertools
import time
import numpy as np
from .common import *
from . import confirm_runtime as rt
from .confirmation_statistics import reconstruct,max_t_intervals,paired_macro_mse_difference
from .report_dev import rank_metrics,discrete_effect

ENDPOINTS=('TRANSFER_300','DAMAGE_300')

def main():
    from .confirm_behavior import cells,directory
    begin=time.monotonic();success={};trials=[];cost=[]
    for cell in cells():
        dest=directory(cell);completion=read(dest/'COMPLETE.json');assert completion['status']=='PASS' and completion['trials']==40
        for f,h in completion['files'].items():assert sha(dest/f)==h
        rows=read(dest/'TRIALS.json');assert sorted(r['trial'] for r in rows)==list(range(40))
        success[cell['seed'],cell['model_id'],cell['task']]=np.array([r['success'] for r in sorted(rows,key=lambda r:r['trial'])]);trials.extend(rows)
        cost.append(dict(kind='behavior',unit=str(cell),seconds=completion['seconds']))
    assert len(trials)==9360;table(rt.REPORT/'ROLLOUTS.csv',trials)
    pairs=[dict(seed=k,source=s,target=t) for k in (48,49,50) for s in range(6) for t in range(6) if s!=t]
    y=reconstruct(pairs,success);pred={};allpred=[];reliability=[];access={}
    frozen=read(OUT/'FINALISTS_FROZEN.json')
    for row in pairs:
        sid=unit(row['seed'],row['source'],row['target']);dest=rt.REPORT/'signals'/sid
        end=read(rt.REPORT/'transfers'/sid/'COMPLETE.json')
        for q in (0,1,5,20):
            sealpath=dest/('C%d_SEALED.json'%q)
            if not sealpath.exists():continue
            seal=read(sealpath);path=dest/('C%d.json'%q);assert sha(path)==seal['signals_sha256']
            assert seal['finalists_sha256']==sha(OUT/'FINALISTS_FROZEN.json') and seal['target_calls_completed']<300
            assert seal['sealed_unix']< (rt.REPORT/'transfers'/sid/'COMPLETE.json').stat().st_mtime
            obj=read(path)
            for pp in obj['predictions']:
                key=(row['seed'],row['source'],row['target'],pp['endpoint'],pp['model'])
                assert key not in pred;pred[key]=pp['prediction'];allpred.append(dict(row,**pp))
            for f in obj['scores']:reliability.append(dict(row,when=q,feature=f,D1=obj['halves'][0].get(f),D2=obj['halves'][1].get(f)))
        for phase in ('static','target'):
            a=read(rt.REPORT/'access'/(phase+'_'+sid+'.json'));assert not any('E_eval' in p or p.endswith(('TARGETS.csv','ROLLOUTS.csv')) or '/behavior/' in p for p in a['paths'])
            access[phase+'_'+sid]=sha(rt.REPORT/'access'/(phase+'_'+sid+'.json'))
        for kind,folder in (('static','signals'),('target','transfers')):
            obj=read(rt.REPORT/folder/sid/('C0.json' if kind=='static' else 'COMPLETE.json'));cost.append(dict(kind=kind,unit=sid,seconds=obj['seconds']))
    table(rt.REPORT/'PREDICTIONS_BEFORE_OUTCOMES.csv',allpred)
    table(rt.REPORT/'SIGNAL_RELIABILITY.csv',reliability)
    write(rt.REPORT/'ACCESS_AUDIT.json',dict(status='PASS',files=access,signals_precede_endpoints=True,formal_training_resumed=False))
    def predictions(e,m):return np.array([pred[r['seed'],r['source'],r['target'],e,m] for r in pairs])
    comparisons=[]
    for c in frozen['candidates']:
        comparisons.append(dict(endpoint=ENDPOINTS.index(c['endpoint']),candidate=predictions(c['endpoint'],c['package']),baseline=predictions(c['endpoint'],c['baseline']),endpoint_name=c['endpoint'],model=c['package'],baseline_name=c['baseline']))
    # One common task/seed index draw is reused across source, reference and all transfers.
    rng=np.random.RandomState(2026092602);rep_y=[];rep_d=[]
    for _ in range(5000):
        indices={(k,t):rng.randint(0,40,40) for k in (48,49,50) for t in range(6)}
        yy=reconstruct(pairs,success,indices);rep_y.append(yy)
        rep_d.append([paired_macro_mse_difference(pairs,yy[:,c['endpoint']],c['candidate'],c['baseline']) for c in comparisons])
    rep_y=np.asarray(rep_y);rep_d=np.asarray(rep_d)
    point=[paired_macro_mse_difference(pairs,y[:,c['endpoint']],c['candidate'],c['baseline']) for c in comparisons]
    intervals=max_t_intervals(point,rep_d);intervals['comparisons']=[dict(model=c['model'],endpoint=c['endpoint_name'],baseline=c['baseline_name']) for c in comparisons]
    write(rt.REPORT/'SIMULTANEOUS_INTERVALS.json',intervals)
    npz(rt.DATA/'statistics/paired_bootstrap.npz',endpoint_values=rep_y,paired_mse_differences=rep_d)
    lo,hi=np.quantile(rep_y,[.025,.975],axis=0);targets=[];uncertainty=[]
    for i,r in enumerate(pairs):
        targets.append(dict(r,**{e:float(y[i,j]) for j,e in enumerate(ENDPOINTS)},same_suite=rt.entry(r['source'])['suite']==rt.entry(r['target'])['suite']))
        for j,e in enumerate(ENDPOINTS):
            a,b=round(float(lo[i,j]),12),round(float(hi[i,j]),12)
            label='POSITIVE' if a>.05 else 'NEGATIVE' if b<-.05 else 'WITHIN_BAND' if a>=-.05 and b<=.05 else 'UNCERTAIN'
            uncertainty.append(dict(r,endpoint=e,lower=a,upper=b,label=label,scope='pointwise paired trial interval, not simultaneous candidate confirmation'))
    table(rt.REPORT/'TARGETS.csv',targets);table(rt.REPORT/'UNCERTAIN_COVERAGE.csv',uncertainty)
    continuous=[];ranking=[];risk=[];sensitivity=[];summaries=[]
    for j,e in enumerate(ENDPOINTS):
        models=sorted({key[-1] for key in pred if key[-2]==e})
        for m in models:
            p=predictions(e,m);yy=y[:,j];slope=float(np.cov(p,yy,bias=True)[0,1]/np.var(p)) if np.ptp(p)>0 else None
            continuous.append(dict(endpoint=e,model=m,mse=float(np.mean((p-yy)**2)),mae=float(np.mean(abs(p-yy))),slope=slope,intercept=float(yy.mean()-slope*p.mean()) if slope is not None else None))
            for fixed in (('target','source') if j==1 else ('target',)):
                for k in (48,49,50):
                    for task in range(6):
                        ix=[i for i,r in enumerate(pairs) if r['seed']==k and r[fixed]==task]
                        ranking.append(dict(endpoint=e,model=m,seed=k,fixed=fixed,task=task,**rank_metrics(np.round(yy[ix],12),p[ix])))
            classes=discrete_effect(yy,40);actual=classes<0 if j==0 else classes>0;warn=p<-.05 if j==0 else p>.05
            tp=int(np.sum(actual&warn));fp=int(np.sum(~actual&warn));fn=int(np.sum(actual&~warn))
            risk.append(dict(endpoint=e,model=m,rows=90,true_positive=tp,false_positive=fp,false_negative=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,false_release=fn,false_suppression=int(np.sum((p<-.05)&(classes>0))) if j==0 else None))
    for index,c in enumerate(comparisons):
        yy=y[:,c['endpoint']];p=c['candidate'];b=c['baseline'];diff=(p-yy)**2-(b-yy)**2
        targetdiff=[float(np.mean([diff[i] for i,r in enumerate(pairs) if r['target']==t])) for t in range(6)]
        seeddiff=[float(np.mean([diff[i] for i,r in enumerate(pairs) if r['seed']==k])) for k in (48,49,50)]
        base=float(np.mean((b-yy)**2));gain=float(-np.mean(diff)/base) if base else None
        for kind,values in [('seed',(48,49,50)),('task',range(6))]:
            for omit in values:
                ix=[i for i,r in enumerate(pairs) if (r['seed']!=omit if kind=='seed' else omit not in (r['source'],r['target']))]
                sensitivity.append(dict(model=c['model'],endpoint=c['endpoint_name'],deleted_kind=kind,deleted_id=omit,rows=len(ix),paired_mse_difference=float(np.mean(diff[ix])),candidate_mse=float(np.mean((p[ix]-yy[ix])**2)),baseline_mse=float(np.mean((b[ix]-yy[ix])**2))))
        gates=dict(relative_gain_at_least5pct=gain is not None and gain>=.05,simultaneous_upper_below_zero=intervals['upper'][index]<0,targets_improved_at_least4=sum(v<0 for v in targetdiff)>=4,more_than_one_seed_improves=sum(v<0 for v in seeddiff)>=2)
        # These are mechanical criteria; task-deletion interpretation gets an explicit final review.
        summaries.append(dict(endpoint=c['endpoint_name'],model=c['model'],baseline=c['baseline_name'],relative_MSE_gain=gain,target_mse_differences=targetdiff,seed_mse_differences=seeddiff,gates=gates,mechanical_criteria_pass=all(gates.values()),final_review_required=True))
    table(rt.REPORT/'CONFIRM_CONTINUOUS.csv',continuous);table(rt.REPORT/'CONFIRM_RANKING.csv',ranking)
    table(rt.REPORT/'CONFIRM_RISK_CLASSIFICATION.csv',risk);table(rt.REPORT/'TASK_SEED_SENSITIVITY.csv',sensitivity)
    # Frozen joint decision; use each endpoint's corresponding candidate role.
    regrets=[]
    selectors=[('BASE','B0','B0')]+[(role,next(c['package'] for c in frozen['candidates'] if c['endpoint']==ENDPOINTS[0] and c['role']==role),next(c['package'] for c in frozen['candidates'] if c['endpoint']==ENDPOINTS[1] and c['role']==role)) for role in ('A_REPLICATION','BEST_NEW_C0','C20')]
    for label,mt,md in selectors:
        pt=predictions(ENDPOINTS[0],mt);pd=predictions(ENDPOINTS[1],md)
        for k in (48,49,50):
            for t in range(6):
                ix=[i for i,r in enumerate(pairs) if r['seed']==k and r['target']==t]
                eligible=[i for i in ix if pt[i]>.05 and pd[i]<=.05]
                chosen=min(eligible,key=lambda i:(-pt[i],pairs[i]['source'])) if eligible else None
                oracle=max([0.]+y[ix,0].tolist());gain=0. if chosen is None else float(y[chosen,0])
                regrets.append(dict(selector=label,seed=k,target=t,source='BASE' if chosen is None else pairs[chosen]['source'],selected_transfer=gain,regret=oracle-gain,base_regret=oracle,uniform_source_expected_regret=oracle-float(y[ix,0].mean())))
    table(rt.REPORT/'DECISION_REGRET.csv',regrets);table(rt.REPORT/'COMPUTE_COST.csv',cost)
    write(rt.REPORT/'STATISTICAL_RESULTS.json',dict(comparisons=summaries,pairs=90,unique_tasks=6,shared_base=True,trials=9360,independent_audit_pending=True,seconds=time.monotonic()-begin,formal_training_resumed=False))
    write(rt.REPORT/'STATUS.json',dict(status='MEASUREMENTS_AND_STATISTICS_COMPLETE_REVIEW_REQUIRED',formal_training_resumed=False,diagnostic_training_performed=True))

if __name__=='__main__':main()
