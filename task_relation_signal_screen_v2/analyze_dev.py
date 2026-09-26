"""Development-only OOF screening. Confirmation outcomes are never consumed."""
import csv
import itertools
import math
import time
import numpy as np
from scipy.stats import spearmanr,kendalltau
from .common import *
from .numeric import ridge

B0=['b_target','b_source','b_progress']
ENDPOINTS=['TRANSFER_300','DAMAGE_300']

def models():
    result={'B0':B0,'B1':B0+['base_progress1'],'B5':B0+['base_progress5'],'B20':B0+['base_progress20']}
    for pkg,fields in PACKAGES.items():
        base=result['B1' if pkg=='S12' else 'B5' if pkg=='S13' else 'B20' if pkg=='S14' else 'B0']
        result[pkg]=base+fields
    # Fixed, limited combined candidates. No cross-products or arbitrary feature search.
    result['S04_S07']=B0+PACKAGES['S04']+PACKAGES['S07']
    result['S14_S04']=result['S14']+PACKAGES['S04']
    return result

def cv(rows,y,features,registry,mode='task',alpha=1.):
    output=np.zeros(len(rows));raw=np.zeros(len(rows));metadata=[]
    for held in (TASKS if mode=='task' else SEEDS):
        tr=[i for i,r in enumerate(rows) if held not in (r['source'],r['target'])] if mode=='task' else [i for i,r in enumerate(rows) if r['seed']!=held]
        te=[i for i,r in enumerate(rows) if r['target']==held] if mode=='task' else [i for i,r in enumerate(rows) if r['seed']==held]
        assert not set(tr)&set(te)
        p,z,m=ridge([rows[i] for i in tr],[rows[i] for i in te],np.asarray(y)[tr],features,registry,alpha)
        output[te]=p;raw[te]=z;metadata.append(dict(held=held,train=tr,test=te,model=m))
    return output,raw,metadata

def mean_mse(rows,y,p):
    return float(np.mean([np.mean([(p[i]-y[i])**2 for i,r in enumerate(rows) if r['target']==t]) for t in sorted(set(r['target'] for r in rows))]))

def nested_dynamic_cv(rows,y,registry,costs,mode='task',alpha=1.):
    """Select the optional C0 in outer TRAIN only; held outcomes never select columns."""
    output=np.zeros(len(rows));raw=np.zeros(len(rows));metadata=[]
    candidates=read(OUT/'SIGNAL_REGISTRY.json')['new_C0_candidates']
    for held in (TASKS if mode=='task' else SEEDS):
        tr=[i for i,r in enumerate(rows) if held not in (r['source'],r['target'])] if mode=='task' else [i for i,r in enumerate(rows) if r['seed']!=held]
        te=[i for i,r in enumerate(rows) if r['target']==held] if mode=='task' else [i for i,r in enumerate(rows) if r['seed']==held]
        rr=[rows[i] for i in tr];yy=np.asarray(y)[tr];inner=[]
        for name in candidates:
            fields=B0+PACKAGES[name]
            if all(all(r.get(f) is None for r in rr) for f in PACKAGES[name]):continue
            pp=np.zeros(len(rr));valid=True
            for node in sorted(set(r['target'] for r in rr)):
                a=[i for i,r in enumerate(rr) if node not in (r['source'],r['target'])]
                b=[i for i,r in enumerate(rr) if r['target']==node]
                if not a or not b:valid=False;break
                pp[b]=ridge([rr[i] for i in a],[rr[i] for i in b],yy[a],fields,registry,alpha)[0]
            if valid:inner.append((mean_mse(rr,yy,pp),costs[name],len(fields),name))
        best=min(inner)[-1] if inner else None
        fields=B0+['base_progress20']+PACKAGES['S14']
        if best and len(PACKAGES[best])<=2:fields=fields+PACKAGES[best]
        p,z,m=ridge(rr,[rows[i] for i in te],yy,fields,registry,alpha)
        output[te]=p;raw[te]=z
        metadata.append(dict(held=held,train=tr,test=te,inner_candidate_scores=inner,training_only_C0=best,fields=fields,model=m))
    return output,raw,metadata

def main():
    begin=time.monotonic();registry=read(OUT/'SIGNAL_REGISTRY.json')['features'];rows=[];reliability=[];steps=[]
    with (OLD/'TARGETS.csv').open() as f:labels={(int(r['seed']),int(r['source']),int(r['target'])):r for r in csv.DictReader(f)}
    for k,s,t in PAIRS:
        static=read(OUT/'static'/unit(k,s,t)/'SIGNALS.json');probe=read(OUT/'probes'/unit(k,s,t)/'MEASUREMENTS.json')
        row=dict(seed=k,source=s,target=t,**static['scores'],**probe['scores'])
        row.update({e:float(labels[k,s,t][e]) for e in ENDPOINTS});rows.append(row)
        for obj in (static,probe):
            for f,v in obj['scores'].items():reliability.append(dict(seed=k,source=s,target=t,feature=f,D1=obj['halves'][0].get(f),D2=obj['halves'][1].get(f),combined=v))
        with (OUT/'probes'/unit(k,s,t)/'PROBE_STEPS.csv').open() as f:
            steps += [dict(seed=k,source=s,target=t,**r) for r in csv.DictReader(f)]
    table(OUT/'SIGNALS_RAW.csv',[{k:v for k,v in r.items() if k not in ENDPOINTS} for r in rows]);table(OUT/'SIGNAL_RELIABILITY.csv',reliability)
    table(OUT/'PROBE_MEASUREMENTS.csv',[{k:v for k,v in r.items() if k in ('seed','source','target') or k.startswith(('risk','base_progress','relative_progress','source_damage','source_tail','damage_'))} for r in rows])
    with (OUT/'PROBE_STEPS.jsonl').open('w') as f:
        import json
        for r in steps:f.write(json.dumps(r,allow_nan=False)+'\n')
    model_defs=models();predictions=[];summary=[];frozen_folds={};scores={};availability={}
    for name,fields in model_defs.items():
        unavailable=[f for f in fields if all(r.get(f) is None for r in rows)]
        extra=PACKAGES.get(name,[f for f in fields if f not in B0 and not f.startswith('base_progress')])
        # A missing component stays null; S10 coverage, for example, remains usable
        # even when conditional action mismatch has insufficient coverage everywhere.
        eligible=not extra or not all(f in unavailable for f in extra)
        availability[name]=dict(fully_unavailable_features=unavailable,eligible=eligible)
    for endpoint in ENDPOINTS:
        y=np.array([r[endpoint] for r in rows]);scores[endpoint]={}
        for name,fields in model_defs.items():
            for mode in ('task','seed'):
                p,raw,meta=cv(rows,y,fields,registry,mode);frozen_folds[endpoint+'/'+name+'/'+mode]=meta
                mse=mean_mse(rows,y,p)
                corr=float(spearmanr(p,y).statistic) if np.ptp(p)>0 else None
                rec=dict(outcome=endpoint,model=name,mode=mode,mse=mse,mae=float(np.mean(abs(p-y))),spearman=corr,available=availability[name]['eligible'])
                for held in (TASKS if mode=='task' else SEEDS):
                    ix=[i for i,r in enumerate(rows) if r['target' if mode=='task' else 'seed']==held]
                    rec['cell%d_mse'%held]=float(np.mean((p[ix]-y[ix])**2))
                summary.append(rec)
                if mode=='task':scores[endpoint][name]=rec
                predictions += [dict(seed=r['seed'],source=r['source'],target=r['target'],outcome=endpoint,model=name,mode=mode,observed=float(y[i]),predicted=float(p[i]),predicted_unclipped=float(raw[i])) for i,r in enumerate(rows)]
        # A constant predictor is fold-local, never the full-data label mean.
        for held in TASKS:
            tr=[i for i,r in enumerate(rows) if held not in (r['source'],r['target'])];value=float(y[tr].mean())
            for i,r in enumerate(rows):
                if r['target']==held:predictions.append(dict(seed=r['seed'],source=r['source'],target=r['target'],outcome=endpoint,model='CONSTANT',mode='task',observed=float(y[i]),predicted=value,predicted_unclipped=value))
    table(OUT/'DEV_OOF_PREDICTIONS.csv',predictions);table(OUT/'DEV_ALL_CANDIDATES.csv',summary);write(OUT/'FOLD_TRANSFORMS.json',frozen_folds)
    write(OUT/'SIGNAL_AVAILABILITY.json',availability)
    # Selection has no p-value prerequisite. Entirely unavailable candidates remain reported.
    chosen=[];selection=[]
    cost_rows=[]
    for path in sorted((OUT/'static').glob('*/COMPUTE_COST.json')):
        item=read(path)
        if item.get('version')!=2:continue
        cost_rows.extend(dict(unit=path.parent.name,package=p,seconds=v,scope=item['scope']) for p,v in item['package_seconds'].items())
    table(OUT/'COMPUTE_COST.csv',cost_rows)
    cost_order={p:float(np.median([r['seconds'] for r in cost_rows if r['package']==p])) for p in PACKAGES if any(r['package']==p for r in cost_rows)}
    assert all(p in cost_order for p in read(OUT/'SIGNAL_REGISTRY.json')['new_C0_candidates']), 'MEASURED_TIE_BREAK_COST_MISSING'
    def select(names,e):
        eligible=[m for m in names if availability[m]['eligible']]
        if not eligible:return None
        return min(eligible,key=lambda m:(scores[e][m]['mse'],sum(cost_order.get(p,0.) for p in m.split('_')),len(model_defs[m]),m))
    for e in ENDPOINTS:
        best=select(read(OUT/'SIGNAL_REGISTRY.json')['new_C0_candidates'],e)
        if best and len(PACKAGES[best])<=2:
            name='S14_'+best;fields=model_defs['S14']+PACKAGES[best]
            model_defs[name]=fields;availability[name]=dict(eligible=availability[best]['eligible'] and availability['S14']['eligible'])
            y=np.array([r[e] for r in rows])
            for mode in ('task','seed'):
                p,raw,meta=nested_dynamic_cv(rows,y,registry,cost_order,mode)
                rec=dict(outcome=e,model=name,mode=mode,mse=mean_mse(rows,y,p),mae=float(np.mean(abs(p-y))),spearman=float(spearmanr(p,y).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else None,available=availability[name]['eligible'])
                summary.append(rec)
                if mode=='task':scores[e][name]=rec
                frozen_folds[e+'/'+name+'/'+mode]=meta
                predictions += [dict(seed=r['seed'],source=r['source'],target=r['target'],outcome=e,model=name,mode=mode,observed=float(y[i]),predicted=float(p[i]),predicted_unclipped=float(raw[i])) for i,r in enumerate(rows)]
        # S14 plus only the selected C0, if the six-extra-feature cap permits.
        dynamic=['S14']+(['S14_'+best] if best is not None and 'S14_'+best in model_defs else [])
        dyn=select(dynamic,e)
        for tag,m in [('A_REPLICATION','S04'),('BEST_NEW_C0',best),('C20',dyn)]:
            if m is None:selection.append(dict(endpoint=e,role=tag,status='UNAVAILABLE'));continue
            y=np.array([r[e] for r in rows]);_,_,fit=ridge(rows,rows,y,model_defs[m],registry)
            chosen.append(dict(endpoint=e,role=tag,package=m,features=model_defs[m],baseline='B20' if m.startswith('S14') else 'B0',fit=fit,development_mse=scores[e][m]['mse'],alpha=1.))
            selection.append(dict(endpoint=e,role=tag,model=m,status='SELECTED_DEVELOPMENT_ONLY'))
    unique=[]
    for c in chosen:
        if not any(a['endpoint']==c['endpoint'] and a['features']==c['features'] for a in unique):unique.append(c)
    baselines=[]
    for e in ENDPOINTS:
        for name in ('B0','B1','B5','B20'):
            _,_,fit=ridge(rows,rows,np.array([r[e] for r in rows]),model_defs[name],registry)
            baselines.append(dict(endpoint=e,model=name,features=model_defs[name],fit=fit))
    freeze(OUT/'FINALISTS_FROZEN.json',dict(status='FROZEN',candidates=unique,baselines=baselines,selection=selection,selection_rule='global C0 selected by development OOF; dynamic-combination OOF reselects C0 using outer-training-only inner task folds; final coefficients use all development rows',tie_break_cost_seconds=cost_order,training_manifest_sha256=sha(OUT/'SIGNALS_RAW.csv'),registry_sha256=sha(OUT/'SIGNAL_REGISTRY.json'),code_sha256=sha(__file__),stage='DEVELOPMENT',no_independent_confirmation_yet=True,formal_training_resumed=False))
    table(OUT/'DEV_OOF_PREDICTIONS.csv',predictions);table(OUT/'DEV_ALL_CANDIDATES.csv',summary);write(OUT/'FOLD_TRANSFORMS.json',frozen_folds)
    # Auxiliary alphas cannot replace the frozen primary choice.
    sensitivity=[]
    for e in ENDPOINTS:
        y=np.array([r[e] for r in rows])
        for name,fields in model_defs.items():
            for alpha in (.1,10.):
                p,_,_=cv(rows,y,fields,registry,alpha=alpha)
                sensitivity.append(dict(endpoint=e,model=name,alpha=alpha,mse=mean_mse(rows,y,p)))
    table(OUT/'ALPHA_SENSITIVITY.csv',sensitivity)
    rankings=[]
    for e in ENDPOINTS:
        for name in model_defs:
            for seed in SEEDS:
                for target in TASKS:
                    rr=[r for r in predictions if r['mode']=='task' and r['outcome']==e and r['model']==name and r['seed']==seed and r['target']==target]
                    a=np.array([r['observed'] for r in rr]);b=np.array([r['predicted'] for r in rr])
                    valid=len(a)>0 and np.ptp(a)>0 and np.ptp(b)>0
                    rankings.append(dict(endpoint=e,model=name,seed=seed,target=target,informative=bool(valid),spearman=float(spearmanr(a,b).statistic) if valid else None,kendall=float(kendalltau(a,b).statistic) if valid else None))
    table(OUT/'DEV_SOURCE_RANKING.csv',rankings)
    lines=['# Stage A：开发筛选结果','', '旧36行、12种有向关系，三个下游优化seed；不构成独立新任务确认。旧闭环未重跑。','', '|终点|角色|入选模型|','|---|---|---|']
    lines+=['|%s|%s|%s|'%(r['endpoint'],r['role'],r.get('model','UNAVAILABLE')) for r in selection]
    lines+=['','全部候选、留任务/留seed结果和原始未裁剪预测分别保存；选择不以开发p值为门。','', '独立任务数据、暴露审计与任务选择尚需完成；尚未启动Stage B，也不宣布任何独立预测价值。','', 'formal_training_resumed=false']
    (OUT/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'DEVELOPMENT_SUMMARY.json',dict(status='MEASURED_AND_FINALISTS_FROZEN',pairs=36,candidate_packages=14,finalist_comparisons=len(unique),negative_control_permutation='PENDING',development_resampling='PENDING',stage_B='NOT_STARTED',seconds=time.monotonic()-begin,formal_training_resumed=False))

if __name__=='__main__':main()
