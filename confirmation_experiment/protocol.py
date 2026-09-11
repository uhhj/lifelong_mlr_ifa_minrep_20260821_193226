"""Frozen protocol and scalar diagnostics. No model or optimizer mutations."""
import math

SEEDS = (45, 46, 47)
ARMS = ('mlr', 'soft')
TASKS = (6, 7, 8, 9)
KAPPA = 0.5
SNAPSHOT_BINS = {'30-69': ((30,43),(43,56),(56,70)),
                 '70-99': ((70,80),(80,90),(90,100))}
SNAPSHOT_MAX = 48
EXPECTED_RECIPE = {
    'benchmark_name':'LIBERO_OBJECT','start_task':6,'end_task':10,
    'past_data_rate':.5,'ifa_enabled':False,'aug':False,
    'repro_scheduler_mode':'batch_step','save_final_checkpoint':True,
}
DECISION_RULE = {
    'minimum_mean_auc_gain_pp':1.0,'minimum_positive_seeds':2,
    'maximum_single_seed_positive_gain_share':0.8,
    'maximum_negative_seed_auc_drop_pp':2.5,
    'systematic_fwt_or_final_drop_pp':2.5,
    'consistent_old_task_drop_pp':5.0,
    'description':'Conservative preregistered descriptive rules, not statistical significance.'
}

def epoch_phase(epoch):
    return '00-09' if epoch<10 else '10-29' if epoch<30 else '30-69' if epoch<70 else '70-99'

def derived_signals(metrics, n_new, n_old):
    """Infer n dot r and ||n|| from g=a*n+b*r; never use z for gating.

    Scalars inherit FP32 gradient/reduction error. No extra backward. Norm
    cancellation is explicit; no silent absolute-value fix of negative squares.
    """
    count=n_new+n_old
    if not count:raise ValueError('Empty batch')
    a,b=n_new/count,n_old/count
    out=dict(a=a,b=b,n_dot_r=None,cos_new_reference=None,z=None,
             new_grad_norm_derived=None,new_norm_cancellation=False,
             n_signal_source='algebra_from_same_forward_mixed_reference_scalars')
    if not (n_new and n_old and metrics['has_reference'] and not metrics['tiny_reference']):return out
    G=metrics['mixed_grad_norm']**2; R=metrics['reference_grad_norm']**2; d=metrics['dot_before']
    nd=(d-b*R)/a
    numerator=G-2*b*d+b*b*R
    scale=G+abs(2*b*d)+b*b*R
    out.update(n_dot_r=nd,z=1-d/(b*R))
    if numerator<=max(1e-20,1e-6*scale):
        out['new_norm_cancellation']=True
    else:
        norm=math.sqrt(numerator)/a
        cos=nd/(norm*metrics['reference_grad_norm'])
        if abs(cos)>1+1e-5:out['new_norm_cancellation']=True
        else:out.update(new_grad_norm_derived=norm,cos_new_reference=max(-1.,min(1.,cos)))
    return out

def snapshot_slot(task,epoch,trigger,z,used,remaining_updates):
    """Three disjoint first-trigger bins + first ordinary-negative control/phase."""
    phase=epoch_phase(epoch)
    if task not in (7,8) or phase not in SNAPSHOT_BINS or remaining_updates<20:return None
    if trigger:
        i=next(i for i,(lo,hi) in enumerate(SNAPSHOT_BINS[phase]) if lo<=epoch<hi)
        slot=f'task{task}_{phase}_trigger_bin{i}'
    elif z is not None and 0<z<=1:
        slot=f'task{task}_{phase}_ordinary_nontrigger'
    else:return None
    return None if slot in used else slot

def decide(deltas,old_task_deltas):
    """New seeds only; no optional stopping and no historical pooling here."""
    if len(deltas)!=3:raise ValueError('All three new paired seeds required')
    avg=lambda k:sum(r[k] for r in deltas)/3
    auc=avg('auc');positive=[r['auc'] for r in deltas if r['auc']>0]
    systematic=any(avg(k)<=-2.5 and sum(r[k]<0 for r in deltas)>=2 for k in ('fwt','final_mean_success'))
    if auc<=0 or systematic:
        return 'FIXED_KAPPA_CONFIRMATION_NOT_SUPPORTED'
    old_decline=any(all(x<0 for x in ds) and sum(ds)/3<=-5 for ds in old_task_deltas.values())
    dominance=max(positive)/sum(positive) if positive else 1.
    if (auc>=1 and len(positive)>=2 and min(r['auc'] for r in deltas)>=-2.5
            and dominance<=.8 and avg('fwt')>=-2.5 and avg('final_mean_success')>=-2.5 and not old_decline):
        return 'FIXED_KAPPA_CONFIRMATION_SUPPORTED'
    return 'FIXED_KAPPA_CONFIRMATION_INCONCLUSIVE'
