"""Independent recomputation of delivered Stage A tables from sealed raw units."""
import csv
import hashlib
import subprocess
import numpy as np
from .common import *
from .numeric import ridge
from .analyze_dev import B0,ENDPOINTS,cv,models

def main():
    checks={};manifest=read(OLD/'DATA_MANIFEST.json');split=read(OUT/'DEMO_SPLITS.json')
    for task in TASKS:
        roles=manifest['tasks'][str(task)]['demos'];s=split[str(task)]
        assert set(s['D1']).isdisjoint(s['D2']) and set(s['D1']+s['D2'])==set(roles['D_signal'])
        assert all(set(roles[a]).isdisjoint(roles[b]) for a,b in [('A_train','D_signal'),('A_train','E_eval'),('D_signal','E_eval')])
    checks['whole_trajectory_roles']=True
    for filename,digest in read(OUT/'SOURCE_PROVENANCE.json')['historical_files'].items():assert sha(OLD/filename)==digest
    checks['historical_results_unchanged']=True
    rows=[]
    with (OLD/'TARGETS.csv').open() as f:labels={(int(r['seed']),int(r['source']),int(r['target'])):r for r in csv.DictReader(f)}
    for k,s,t in PAIRS:
        a=read(OUT/'static'/unit(k,s,t)/'SIGNALS.json');b=read(OUT/'probes'/unit(k,s,t)/'MEASUREMENTS.json')
        assert all(a['state_unchanged'].values()) and a['repeat_nll_abs_max']==0
        for obj in (a,b):
            for key,value in obj['scores'].items():
                halves=[q.get(key) for q in obj['halves']]
                if any(v is None for v in halves):assert value is None
                else:assert value==float(np.mean(halves))
        for kind in ('static','probe'):
            access=read(OUT/'access'/('%s_%s.json'%(kind,unit(k,s,t))))
            assert not access['E_eval_accessed'] and not access['outcomes_accessed']
        with (OUT/'probes'/unit(k,s,t)/'PROBE_STEPS.csv').open() as f:steps=list(csv.DictReader(f))
        for branch in ('source','base'):
            rr=[r for r in steps if r['branch']==branch]
            assert float(rr[0]['lr'])==0 and float(rr[0]['d_norm'])==0
            effective=0
            for call,r in enumerate(rr,1):
                effective+=int(float(r['lr'])>0 and float(r['d_norm'])>0)
                assert int(r['call'])==int(r['optimizer_step'])==int(r['scheduler_step'])==call
                assert int(r['effective_updates'])==effective
                assert abs(float(r['loss'])-float(r['original_loss']))<=max(1e-6,1e-6*abs(float(r['original_loss'])))
        rows.append(dict(seed=k,source=s,target=t,**a['scores'],**b['scores']))
    checks.update(raw_half_aggregation=True,shadow_states_preserved=True,zero_lr_negative_control=True,prefix_history_parity=True,signal_access_firewall=True)
    with (OUT/'SIGNALS_RAW.csv').open() as f:published=list(csv.DictReader(f))
    assert len(published)==len(rows)==36
    for a,b in zip(rows,published):
        for key,value in a.items():assert (b[key]=='' if value is None else float(b[key])==value),(key,b[key],value)
    registry=read(OUT/'SIGNAL_REGISTRY.json')['features']
    with (OUT/'DEV_OOF_PREDICTIONS.csv').open() as f:pred=list(csv.DictReader(f))
    maximum=0.
    for endpoint in ENDPOINTS:
        y=np.array([float(labels[r['seed'],r['source'],r['target']][endpoint]) for r in rows])
        for name,features in models().items():
            for mode in ('task','seed'):
                actual,_,_=cv(rows,y,features,registry,mode)
                saved=[r for r in pred if r['outcome']==endpoint and r['model']==name and r['mode']==mode]
                assert len(saved)==36
                difference=float(np.max(abs(actual-np.array([float(r['predicted']) for r in saved]))));maximum=max(maximum,difference)
                assert difference<=1e-12
    checks['oof_numeric_recomputation']=True
    finalists=read(OUT/'FINALISTS_FROZEN.json');assert len(finalists['candidates'])<=6
    for candidate in finalists['candidates']:
        y=np.array([float(labels[r['seed'],r['source'],r['target']][candidate['endpoint']]) for r in rows])
        _,_,fit=ridge(rows,rows,y,candidate['features'],registry)
        assert fit==candidate['fit']
    checks['finalist_coefficient_recomputation']=True
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'task_relation_signal_screen_v2').rglob('*.py'))}
    write(OUT/'INDEPENDENT_STAGE_A_AUDIT.json',dict(status='PASS',checks=checks,maximum_prediction_recalculation_difference=maximum,source_files=files,units=36,independent_confirmation_complete=False,formal_training_resumed=False))
    print('INDEPENDENT_STAGE_A_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
