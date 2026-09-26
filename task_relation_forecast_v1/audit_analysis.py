"""Independent raw-log and per-trajectory audit plus declared sensitivity tables."""
import collections
import csv
import json
import time
import numpy as np
from scipy.stats import spearmanr
from .common import OUT, TASKS, SEEDS, read, write
from .statistics import FAMILIES, predict_fold
from .analyze import load_csv, table, independent_ridge, metric


def main():
    begin=time.monotonic();data=read(OUT/'DATA_MANIFEST.json');audit={}
    for task in TASKS:
        d=data['tasks'][str(task)]['demos'];a,s,e=map(set,[d['A_train'],d['D_signal'],d['E_eval']])
        assert not (a&s or a&e or s&e)
    nll_errors=[];window_count=0;seen={};stream_checks=[]
    for seed in SEEDS:
        for source in TASKS:
            for target in TASKS:
                if source==target:continue
                unit='s%d_%d_%d'%(seed,source,target);path=OUT/'offline'/unit
                groups=collections.defaultdict(list)
                for r in load_csv(path/'WINDOW_NLL.csv'):
                    task=int(r['eval_task']);assert r['trajectory_id'] in data['tasks'][str(task)]['demos']['E_eval']
                    groups[r['model'],int(r['step']),task,r['trajectory_id']].append(float(r['nll']));window_count+=1
                for r in load_csv(path/'OFFLINE_NLL.csv'):
                    model,step,task=r['model'],int(r['step']),int(r['eval_task'])
                    values=[np.mean(v) for k,v in groups.items() if k[:3]==(model,step,task)]
                    assert len(values)==10
                    actual=float(np.mean(values));err=abs(actual-float(r['mean_trajectory_nll']));nll_errors.append(err)
                    key=(seed,r['model_sha256'],task)
                    if key in seen:assert abs(actual-seen[key])<1e-12
                    seen[key]=actual
                tp=OUT/'transfers'/unit;source_prefix=OUT/'sources'/('s%d_t%d'%(seed,target))
                for i in range(1,301):
                    x=read(tp/'steps'/('%04d.json'%i));y=read(source_prefix/'steps'/('%04d.json'%i))
                    for k in ['record_indices','lr','lr_after','optimizer_step','scheduler_step']:assert x[k]==y[k],(unit,i,k)
                stream_checks.append(dict(unit=unit,matched_steps=300))
    assert max(nll_errors)<1e-12
    rows=[{k:(int(v) if k in ['seed','source','target'] else float(v)) for k,v in r.items()} for r in load_csv(OUT/'SIGNALS.csv')]
    targets=load_csv(OUT/'TARGETS.csv');lookup={(int(r['seed']),int(r['source']),int(r['target'])):r for r in targets}
    # Independent label calculation reads original trials, not production reconstruction.
    label_error=0.
    for r in rows:
        seed,s,t=r['seed'],r['source'],r['target']
        def p(model,task):return sum(int(v['success']) for v in read(OUT/'behavior'/('s%d'%seed)/model/('task%d'%task)/'TRIALS.json'))/20.
        t0=p('source%d'%s,t)-p('base',t);transfer=p('transfer%d_%d'%(s,t),t)-p('reference%d'%t,t)
        expected=dict(TRANSFER_0=t0,TRANSFER_300=transfer,ACQUISITION_DIFF_300=transfer-t0,DAMAGE_300=p('source%d'%s,s)-p('transfer%d_%d'%(s,t),s))
        for k,v in expected.items():label_error=max(label_error,abs(v-float(lookup[seed,s,t][k])))
    assert label_error<1e-15
    sensitivity=[];max_ridge_error=0.;max_ridge_relative_error=0.;max_prediction_abs=0.
    for removed in TASKS:
        rr=[r for r in rows if removed not in (r['source'],r['target'])]
        for outcome in ['TRANSFER_300','DAMAGE_300']:
            y=np.array([float(lookup[r['seed'],r['source'],r['target']][outcome]) for r in rr])
            for family in FAMILIES:
                pred=np.full(len(rr),np.nan)
                for held in TASKS:
                    if held==removed:continue
                    train=[i for i,r in enumerate(rr) if held not in (r['source'],r['target'])]
                    test=[i for i,r in enumerate(rr) if r['target']==held]
                    p,_=predict_fold(rr,family,y,train,test);pred[test]=p
                    ind=independent_ridge(rr,family,y,train,test);max_ridge_error=max(max_ridge_error,float(np.max(abs(ind-p))))
                    max_ridge_relative_error=max(max_ridge_relative_error,float(np.max(abs(ind-p)/np.maximum(1,abs(p)))))
                    max_prediction_abs=max(max_prediction_abs,float(np.max(abs(p))))
                assert np.isfinite(pred).all()
                sensitivity.append(dict(removed_task=removed,outcome=outcome,family=family,**metric(y,pred)))
    assert max_ridge_relative_error<1e-12
    table(OUT/'TASK_DELETION_SENSITIVITY.csv',sensitivity)
    # Single-feature association is descriptive, never replaces frozen predictor families.
    associations=[]
    for signal in ['A','S_lat','S_GN','S_visual','S_language','R_old','R_new']:
        x=np.array([r[signal] for r in rows])
        for outcome in ['TRANSFER_0','TRANSFER_300','ACQUISITION_DIFF_300','DAMAGE_300']:
            y=np.array([float(lookup[r['seed'],r['source'],r['target']][outcome]) for r in rows]);defined=np.ptp(x)>0 and np.ptp(y)>0
            associations.append(dict(signal=signal,outcome=outcome,spearman=float(spearmanr(x,y).statistic) if defined else None,status='DESCRIPTIVE' if defined else 'UNDEFINED_CONSTANT'))
    write(OUT/'DESCRIPTIVE_ASSOCIATIONS.json',associations)
    audit=dict(status='PASS',offline_windows_recomputed=window_count,trajectory_nll_max_error=max(nll_errors),unique_endpoint_task_nll=len(seen),data_trajectory_disjoint=True,target_reference_stream_checks=stream_checks,raw_trial_label_max_error=label_error,task_deletion_independent_ridge_max_error=max_ridge_error,task_deletion_independent_ridge_max_relative_error=max_ridge_relative_error,task_deletion_max_abs_prediction=max_prediction_abs,sensitivity_limitation='Near-constant nonzero input similarity can amplify extrapolation in the reduced two-task training folds. Original exact-zero rule and unclipped ridge predictions retained. This is a sensitivity failure, not a reason to tune variance thresholds.',seconds=time.monotonic()-begin,formal_training_resumed=False)
    write(OUT/'RAW_DATA_INDEPENDENT_AUDIT.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
