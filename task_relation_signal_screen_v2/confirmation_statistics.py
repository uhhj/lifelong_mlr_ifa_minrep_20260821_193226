"""Frozen-panel paired-trial reconstruction and simultaneous max-T intervals.

No classifier fitting occurs here. This utility does not start confirmation.
"""
import numpy as np

def reconstruct(rows,success,indices=None):
    def rate(seed,model,task):
        value=np.asarray(success[seed,model,task],dtype=float)
        assert value.ndim==1 and len(value)>0 and np.isin(value,[0,1]).all()
        return float(np.mean(value if indices is None else value[indices[seed,task]]))
    values=[]
    for row in rows:
        k,s,t=row['seed'],row['source'],row['target'];model='transfer%d_%d'%(s,t)
        values.append([rate(k,model,t)-rate(k,'reference%d'%t,t),rate(k,'source%d'%s,s)-rate(k,model,s)])
    return np.asarray(values)

def paired_macro_mse_difference(rows,y,candidate,baseline):
    difference=(np.asarray(candidate)-y)**2-(np.asarray(baseline)-y)**2
    return float(np.mean([np.mean(difference[[i for i,r in enumerate(rows) if r['target']==t]]) for t in sorted(set(r['target'] for r in rows))]))

def max_t_intervals(point,replicates,confidence=.95):
    point=np.asarray(point,dtype=float);reps=np.asarray(replicates,dtype=float)
    assert reps.ndim==2 and reps.shape[1]==len(point) and np.isfinite(reps).all()
    se=reps.std(0,ddof=1);active=se>0
    # Center at bootstrap mean; fixed frozen prediction panel, simultaneous across comparisons.
    centered=reps-reps.mean(0)
    maximum=np.max(abs(centered[:,active]/se[active]),axis=1) if active.any() else np.zeros(len(reps))
    critical=float(np.quantile(maximum,confidence))
    return dict(point=point.tolist(),standard_error=se.tolist(),critical=critical,
                lower=(point-critical*se).tolist(),upper=(point+critical*se).tolist(),
                zero_variance=(~active).tolist(),replicates=len(reps),confidence=confidence,
                scope='fixed tasks, fixed models, shared initial-state evaluation uncertainty; not task-population inference')

def paired_trials(rows,success,predictions,count=5000,seed=2026092602):
    """predictions: sequence of endpoint(0/1),candidate,baseline frozen arrays."""
    keys=sorted(set((r['seed'],t) for r in rows for t in (r['source'],r['target'])))
    sizes={key:len(next(v for (k,model,t),v in success.items() if (k,t)==key)) for key in keys}
    assert all(len(v)==sizes[k,t] for (k,model,t),v in success.items())
    def metric(y):
        return [paired_macro_mse_difference(rows,y[:,p['endpoint']],p['candidate'],p['baseline']) for p in predictions]
    rng=np.random.RandomState(seed);point=metric(reconstruct(rows,success));reps=[]
    for _ in range(count):
        indices={key:rng.randint(0,n,n) for key,n in sizes.items()}
        reps.append(metric(reconstruct(rows,success,indices)))
    return max_t_intervals(point,reps),np.asarray(reps)
