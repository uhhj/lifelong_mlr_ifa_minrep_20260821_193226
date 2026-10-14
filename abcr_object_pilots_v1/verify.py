"""Independent checks from saved arrays/tables; no encoder or optimization."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from .common import sha,write

def verify(out):
    out=Path(out);started=time.perf_counter();manifest=json.loads((out/'DATA_MANIFEST.json').read_text())
    protocol=json.loads((out/'PROTOCOL.json').read_text());norm=json.loads((out/'NORMALIZATION.json').read_text())
    assert manifest['protocol_sha256']==sha(out/'PROTOCOL.json')
    checked_sources=[]
    for asset in manifest['assets']:
        assert sha(asset['path'])==asset['sha256'], 'RAW_INPUT_CHANGED: '+asset['path']
        checked_sources.append(asset['sha256'])
    raw={};tasks=[];demos=[]
    for rec in manifest['demos']:
        assert sha(out/(rec['id']+'_raw.npz'))==rec['raw_sha256']
        comp=json.loads((out/(rec['id']+'_FEATURE_COMPLETE.json')).read_text())
        assert sha(out/(rec['id']+'_features.npz'))==comp['sha256']
        with np.load(out/(rec['id']+'_raw.npz')) as z:
            for key in ['future','future_valid','eef','target_relative','target_valid']:
                raw.setdefault(key,[]).append(z[key])
        tasks.extend([rec['task']]*rec['n']);demos.extend([rec['demo']]*rec['n'])
    raw={k:np.concatenate(v) for k,v in raw.items()};tasks=np.array(tasks);demos=np.array(demos)
    scale=np.array(norm['action_scale']);checks=[];max_action_error=0.;retrieval_rows=0
    for task in [6,7,8,9]:
        stage=out/('task%d'%task);completion=json.loads((stage/'COMPLETE.json').read_text())
        for name,digest in completion['outputs'].items(): assert sha(stage/name)==digest
        p1=json.loads((stage/'P1.json').read_text());p3=json.loads((stage/'P3.json').read_text())
        contrib=json.loads((stage/'BY_OLD_TASK.json').read_text())
        ni=np.where(tasks==task)[0];oi=np.where((tasks<task)&(demos<5))[0]
        ds=np.memmap(stage/'Ds.float64',mode='r',dtype='float64',shape=(len(ni),len(oi)))
        ca=np.memmap(stage/'cos_agent.float64',mode='r',dtype='float64',shape=ds.shape)
        ce=np.memmap(stage/'cos_eih.float64',mode='r',dtype='float64',shape=ds.shape)
        for row in p1:
            beta=row['beta'];score=beta*np.asarray(ce)+(1-beta)*np.asarray(ca)
            assert float(np.quantile(ds,.15))==row['tau_s']
            assert float(np.quantile(score,.85))==row['tau_v']
            i,j=np.where((ds<=row['tau_s'])&(score>=row['tau_v']))
            assert len(i)==row['comparable_pairs']
            assert len(np.unique(i))/len(ni)==row['state_anchor_coverage']
            ok=raw['future_valid'][ni[i],:4].all(1)&raw['future_valid'][oi[j],:4].all(1)
            assert int(ok.sum())==row['comparable_q4_pairs']
            assert sum(r['share_pairs'] for r in contrib if r['beta']==beta)==row['share_pairs']
            assert sum(r['bif_pairs'] for r in contrib if r['beta']==beta)==row['bif_pairs']
            with np.load(stage/('P3_common_beta%.1f.npz'%beta)) as z:
                indices=np.unique(np.linspace(0,len(z['new_index'])-1,min(101,len(z['new_index'])),dtype=int))
                for n in indices:
                    ii=int(z['new_index'][n]);jj=int(z['old_index'][n])
                    for q in range(1,7):
                        total=0.
                        for h in range(q):
                            for k in range(7):
                                diff=(float(raw['future'][ii,h,k])-float(raw['future'][jj,h,k]))/float(scale[k])
                                total+=diff*diff
                        direct=(total/(7*q))**.5
                        error=abs(direct-float(z['Dq%d'%q][n]));max_action_error=max(max_action_error,error)
                        assert error<1e-12
                for q in range(1,7):
                    record=next(r for r in p3 if r['beta']==beta and r['q']==q)
                    shared=z['Dq1']<=row['tau_plus']
                    assert int(shared.sum())==record['q1_shared_count']
                    assert int((shared&(z['Dq%d'%q]>=row['tau_minus'])).sum())==record['q1_shared_now_bif_count']
            with open(stage/('P2_retrieval_beta%.1f.csv'%beta),newline='') as f:
                for r in csv.DictReader(f):
                    ii=int(r['new_index']);jj=int(r['old_index']);retrieval_rows+=1
                    if jj<0: assert r['eef_error_m']=='';continue
                    assert tasks[ii]==task and tasks[jj]<task and demos[jj]<5
                    error=sum((float(a)-float(b))**2 for a,b in zip(raw['eef'][ii],raw['eef'][jj]))**.5
                    assert abs(error-float(r['eef_error_m']))<1e-12
                    if raw['target_valid'][ii] and raw['target_valid'][jj]:
                        error=sum((float(a)-float(b))**2 for a,b in zip(raw['target_relative'][ii],raw['target_relative'][jj]))**.5
                        assert abs(error-float(r['object_relative_vector_error_m']))<1e-12
                    else: assert r['object_relative_vector_error_m']==''
            checks.append(dict(task=task,beta=beta,quantiles_exact=True,counts_consistent=True))
            del score
        del ds,ca,ce
    result=dict(status='PASS',checks=checks,original_input_assets=len(checked_sources),demos=len(manifest['demos']),
        p2_all_retrieval_rows=retrieval_rows,p3_scalar_recompute_max_abs=max_action_error,
        seconds=time.perf_counter()-started,code_sha256=sha(__file__),
        limitation='Same CPU environment; arithmetic implementation cross-check, not an independent model re-export.',formal_training_resumed=False)
    write(out/'INDEPENDENT_CHECK.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);verify(p.parse_args().out)
