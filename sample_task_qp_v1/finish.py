"""Independent checks from per-point raw files, no production analyzer imports."""
import csv,numpy as np
from .common import *
def run():
    protocol=read(OUT/'PROTOCOL_FROZEN.json')
    with (OUT/'summary.csv').open() as f:reported=list(csv.DictReader(f))
    raw_summary={};n=0
    for point in protocol['points']:
        d=OUT/'measurements'/point;c=read(d/'COMPLETE.json')
        for fn,h in c['files'].items():assert sha(d/fn)==h
        dec=read(d/'DECISIONS_FROZEN.json');G=read(d/'SAMPLE_GRAM.json');K=np.asarray(G['gram']);assert len(G['rows']) in [32,64]
        assert all(len([x for x in G['rows'] if x['eval_task']==t])==32 for t in {x['eval_task'] for x in G['rows']})
        for arm in ['SAMPLE_GRAD_QP','SAMPLE_STEP_QP']:
            s=dec['arms'][arm];lam=np.asarray(s['lam']);f=np.asarray(s['f']);r=f-K@lam;valid=np.asarray(s['valid'],dtype=int);den=np.maximum(np.sqrt(np.diag(K))*s['u_norm'],1e-30)
            assert np.min(lam)>=0 and np.max(r[valid]/den[valid],initial=0)<=1e-8
            assert abs(float(lam@K@lam)-s['correction_norm']**2)<=1e-7*max(1,float(lam@K@lam))
        for arm in ARMS:
            state=read(d/(arm+'_STATE_HASHES.json'));native=read(d/'MLR_STATE_HASHES.json')
            for key in ['rng','scheduler','optimizer_step','scheduler_step','training_modes']:assert state[key]==native[key]
            if 'STEP' in arm:assert state['optimizer']==native['optimizer']
        vals=rows(d/'AB_WINDOWS.jsonl');updates=rows(d/'UPDATE_METRICS.jsonl')
        for arm in ARMS:
            u=next(x for x in updates if x['arm']==arm)
            for bank in ['A','B']:
                a=[x for x in vals if x['arm']==arm and x['bank']==bank];tasks=sorted({x['eval_task'] for x in a});current=a[0]['current_task'];dd={}
                for t in tasks:
                    samples=[x for x in a if x['eval_task']==t]
                    assert all(abs((x['nll']-x['MLR_nll'])-x['delta_vs_MLR'])<1e-12 for x in samples)
                    dd[t]=sum(x['nll']-x['MLR_nll'] for x in samples)/len(samples)
                old=[dd[t] for t in tasks if t<current]
                entry=(sum(old)/len(old),dd[current],u['actual_norm_ratio'])
                for seed in [str(a[0]['seed']),'ALL']:raw_summary.setdefault((seed,arm,bank),[]).append(entry)
        n+=1
    for row in reported:
        if row['bank']=='P':continue
        xs=raw_summary[row['seed'],row['arm'],row['bank']]
        assert len(xs)==int(row['origins'])
        for i,col in enumerate(['old_balanced_delta','new_delta','actual_norm_ratio']):assert abs(float(row[col])-sum(x[i] for x in xs)/len(xs))<1e-10
    assert n==36
    write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',points=n,raw_sample_QP_KKT=True,optimizer_state_rules=True,summary_recomputed_without_analyzer=True))
    write(OUT/'DECISION.json',dict(status='ONE_STEP_DIAGNOSTIC_COMPLETE',no_closed_loop=True,no_persistence_claim=True,controller_effectiveness_not_decided=True))
    print('INDEPENDENT_AUDIT_PASS',n)
if __name__=='__main__':run()
