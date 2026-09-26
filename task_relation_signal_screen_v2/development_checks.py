"""Fixed 1000 blocked permutations and development block bootstrap; not confirmation."""
import csv
import itertools
import time
import numpy as np
from .common import *
from .analyze_dev import cv,mean_mse,B0,ENDPOINTS

def main():
    dest=OUT/'development_checks';dest.mkdir(exist_ok=True)
    with (OUT/'SIGNALS_RAW.csv').open() as f:
        rows=[{k:(int(v) if k in ('seed','source','target') else float(v) if v else None) for k,v in r.items()} for r in csv.DictReader(f)]
    with (OLD/'TARGETS.csv').open() as f:labels={(int(r['seed']),int(r['source']),int(r['target'])):r for r in csv.DictReader(f)}
    for r in rows:r.update({e:float(labels[r['seed'],r['source'],r['target']][e]) for e in ENDPOINTS})
    registry=read(OUT/'SIGNAL_REGISTRY.json')['features'];pairs=list(itertools.combinations(TASKS,2));lookup={(r['seed'],r['source'],r['target']):r for r in rows}
    defs={name:B0+(['base_progress1'] if name=='S12' else ['base_progress5'] if name=='S13' else ['base_progress20'] if name=='S14' else [])+fields for name,fields in PACKAGES.items()}
    actual={};actual_baseline={}
    baseline_defs={'B0':B0,'B1':B0+['base_progress1'],'B5':B0+['base_progress5'],'B20':B0+['base_progress20']}
    baseline_for=lambda name:'B1' if name=='S12' else 'B5' if name=='S13' else 'B20' if name=='S14' else 'B0'
    for e in ENDPOINTS:
        y=np.array([r[e] for r in rows])
        for name,fields in defs.items():actual[e+'/'+name]=mean_mse(rows,y,cv(rows,y,fields,registry)[0])
        for name,fields in baseline_defs.items():actual_baseline[e+'/'+name]=mean_mse(rows,y,cv(rows,y,fields,registry)[0])
    for rep in range(1000):
        path=dest/('%04d.json'%rep)
        if path.exists():continue
        rng=np.random.RandomState(stable_seed('screen-v2-development-checks',rep))
        mapping=dict(zip(pairs,[pairs[i] for i in rng.permutation(len(pairs))]))
        perm={};boot={};boot_diff={}
        for e in ENDPOINTS:
            y=np.array([r[e] for r in rows])
            for name,fields in defs.items():
                shuffled=[]
                for r in rows:
                    pair=tuple(sorted((r['source'],r['target'])));mapped=mapping[pair]
                    a,b=mapped if r['source']<r['target'] else mapped[::-1]
                    donor=lookup[r['seed'],a,b]
                    shuffled.append(dict(r,**{f:donor[f] for f in PACKAGES[name]}))
                perm[e+'/'+name]=mean_mse(shuffled,y,cv(shuffled,y,fields,registry)[0])
        sampled_seeds=rng.choice(SEEDS,len(SEEDS),replace=True);sampled_pairs=[pairs[i] for i in rng.randint(len(pairs),size=len(pairs))]
        sample=[lookup[int(k),a,b] for k in sampled_seeds for pair in sampled_pairs for a,b in (pair,pair[::-1])]
        valid=all(any(r['target']==held for r in sample) and any(held not in (r['source'],r['target']) for r in sample) for held in TASKS)
        if valid:
            for e in ENDPOINTS:
                yy=np.array([r[e] for r in sample])
                bm={name:mean_mse(sample,yy,cv(sample,yy,fields,registry)[0]) for name,fields in baseline_defs.items()}
                for name,fields in defs.items():
                    boot[e+'/'+name]=mean_mse(sample,yy,cv(sample,yy,fields,registry)[0])
                    boot_diff[e+'/'+name]=boot[e+'/'+name]-bm[baseline_for(name)]
        write(path,dict(rep=rep,permutation_mse=perm,bootstrap_mse=boot,bootstrap_paired_mse_difference=boot_diff,bootstrap_all_task_folds_available=valid))
        if rep%25==0:write(OUT/'DEVELOPMENT_CHECK_STATUS.json',dict(completed=rep+1,budget=1000,formal_training_resumed=False))
    allr=[read(dest/('%04d.json'%r)) for r in range(1000)];records=[]
    for key,value in actual.items():
        pv=np.array([r['permutation_mse'][key] for r in allr]);bv=np.array([r['bootstrap_mse'][key] for r in allr if key in r['bootstrap_mse']])
        dv=np.array([r['bootstrap_paired_mse_difference'][key] for r in allr if key in r['bootstrap_mse']]);e,name=key.split('/')
        records.append(dict(key=key,observed_mse=value,matched_baseline=baseline_for(name),baseline_mse=actual_baseline[e+'/'+baseline_for(name)],observed_paired_mse_difference=value-actual_baseline[e+'/'+baseline_for(name)],permutation_mse_mean=float(pv.mean()),permutation_lower_tail_fraction=float((1+sum(pv<=value))/1001),development_bootstrap_valid=len(bv),development_bootstrap_q025=float(np.quantile(bv,.025)) if len(bv) else None,development_bootstrap_q975=float(np.quantile(bv,.975)) if len(bv) else None,paired_difference_q025=float(np.quantile(dv,.025)) if len(dv) else None,paired_difference_q975=float(np.quantile(dv,.975)) if len(dv) else None))
    table(OUT/'DEVELOPMENT_NEGATIVE_CONTROLS.csv',records)
    write(OUT/'DEVELOPMENT_RESAMPLING.json',dict(status='COMPLETE',replicates=1000,seed='SHA256 stable_seed(screen-v2-development-checks,rep)',permutation='unordered pair block permutation shared across all seeds; directional order preserved; only package shuffled, baselines unchanged',bootstrap='3 seed blocks and 6 unordered pair blocks sampled with replacement; refit every held-task fold; no redraw for missing folds',incomplete_fold_replicates=sum(not r['bootstrap_all_task_folds_available'] for r in allr),scope='development descriptive only, not Stage B simultaneous confirmation intervals',formal_training_resumed=False))
    summary=read(OUT/'DEVELOPMENT_SUMMARY.json');summary.update(negative_control_permutation='COMPLETE',development_resampling='COMPLETE');write(OUT/'DEVELOPMENT_SUMMARY.json',summary)

if __name__=='__main__':main()
