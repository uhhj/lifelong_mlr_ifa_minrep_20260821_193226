import csv,collections,statistics
import numpy as np
from .common import *
def paired(trials):
    d=collections.defaultdict(dict)
    for x in trials:
        key=(x['point'],x['task'],x['trial']);assert x['arm'] not in d[key];d[key][x['arm']]=x
    assert all(set(v)==set(ARMS) for v in d.values()),'missing paired trial'
    return d
def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    verify();p=read(OUT/'PROTOCOL_FROZEN.json');allrows=[];checks=[]
    for c in p['cells']:
        d=OUT/'evaluation'/c['point']/('task'+str(c['task']))/c['arm'];x=read(d/'COMPLETE.json')
        for fn,h in x['files'].items():assert sha(d/fn)==h
        rr=read(d/'TRIALS.json');assert len(rr)==20 and [x['trial'] for x in rr]==list(range(20));allrows+=rr
        for j,r in enumerate(read(d/'ROUNDS.json')):
            with np.load(d/f'round{j}.npz') as z:
                derived=trial_summary(z['done'],z['steps'])
                assert [v['success'] for v in derived]==r['success']
                assert [v['first_success_step'] for v in derived]==r['first']
            checks.append({'point':c['point'],'task':c['task'],'arm':c['arm'],'round':j,'pass':True})
    pairs=paired(allrows);assert len(pairs)==1800
    cellrows=[];originrows=[];seedrows=[];taskrows=[]
    for name in p['points']:
        seed=int(name.split('_')[0][1:]);current=int(name.split('_')[1][1:]);tasks=p['tasks_by_point'][name]
        byarm={a:{} for a in ARMS}
        for t in tasks:
            pd=[pairs[name,t,i] for i in range(20)]
            ds=[OUT/'evaluation'/name/('task'+str(t))/a for a in ARMS]
            r0,r1=[read(x/'ROUNDS.json') for x in ds]
            for j in range(2):
                for k in ['init_sha256','initial_obs_sha256','policy_seed','policy_rng_before']:assert r0[j][k]==r1[j][k],('paired eval input mismatch',name,t,j,k)
            for a in ARMS:byarm[a][t]=sum(v[a]['success'] for v in pd)*5.
            wins=sum(v[ARMS[1]]['success']>v[ARMS[0]]['success'] for v in pd);losses=sum(v[ARMS[1]]['success']<v[ARMS[0]]['success'] for v in pd)
            cellrows.append({'point':name,'seed':seed,'current_task':current,'eval_task':t,'MLR':byarm[ARMS[0]][t],'TASK_GRAD_QP':byarm[ARMS[1]][t],'delta_pp':byarm[ARMS[1]][t]-byarm[ARMS[0]][t],'paired_wins':wins,'paired_losses':losses,'ties':20-wins-losses})
        row={'point':name,'seed':seed,'current_task':current,'epoch':int(name.split('_')[2][1:])}
        for a in ARMS:
            row[a+'_seen']=statistics.mean(byarm[a].values());row[a+'_old']=statistics.mean(v for t,v in byarm[a].items() if t<current);row[a+'_new']=byarm[a][current];row[a+'_worst_old']=min(v for t,v in byarm[a].items() if t<current)
        for m in ['seen','old','new','worst_old']:row['delta_'+m]=row[ARMS[1]+'_'+m]-row[ARMS[0]+'_'+m]
        originrows.append(row)
    for seed in p['seeds']:
        rr=[r for r in originrows if r['seed']==seed];assert len(rr)==12
        row={'seed':seed}
        for a in ARMS:
            for m in ['seen','old','new','worst_old']:row[a+'_'+m]=statistics.mean(r[a+'_'+m] for r in rr)
        for m in ['seen','old','new','worst_old']:row['delta_'+m]=statistics.mean(r['delta_'+m] for r in rr)
        row['origins_better']=sum(r['delta_seen']>0 for r in rr);row['origins_worse']=sum(r['delta_seen']<0 for r in rr)
        seedrows.append(row)
        for task in [6,7,8]:
            vv=[r for r in cellrows if r['seed']==seed and r['eval_task']==task]
            taskrows.append({'seed':seed,'task':task,'origins':len(vv),'MLR':statistics.mean(r['MLR'] for r in vv),'TASK_GRAD_QP':statistics.mean(r['TASK_GRAD_QP'] for r in vv),'delta_pp':statistics.mean(r['delta_pp'] for r in vv)})
    rng=np.random.RandomState(20260915);boot=[]
    for _ in range(10000):
        boot.append(np.mean([np.mean(rng.choice([r['delta_seen'] for r in originrows if r['seed']==s],12,replace=True)) for s in p['seeds']]))
    summary={k:statistics.mean(r[k] for r in seedrows) for k in seedrows[0] if k!='seed'}
    criteria={'primary_practical_gain':summary['delta_seen']>=2.5,'at_least_two_positive_seeds':sum(r['delta_seen']>0 for r in seedrows)>=2,'no_seed_primary_drop_5pp':all(r['delta_seen']>-5 for r in seedrows),'old_new_pooled_guard':all(summary['delta_'+m]>=-2.5 for m in ['old','new']),'old_new_seed_guard':all(r['delta_'+m]>-5 for r in seedrows for m in ['old','new']),'no_systematic_task_drop_10pp':all(sum(r['delta_pp']<=-10 for r in taskrows if r['task']==t)<2 for t in [6,7,8])}
    decision='CANDIDATE_FOR_FULL_TRAINING_STUDY' if all(criteria.values()) else ('NOT_SUPPORTED_ON_THIS_PROTOCOL' if summary['delta_seen']<=0 else 'INCONCLUSIVE')
    # Independent count-based recomputation without production cell/origin aggregators.
    independent={}
    for seed in p['seeds']:
        vals=[]
        for name in [n for n in p['points'] if n.startswith('s'+str(seed)+'_')]:
            vals.append(sum((sum(x['success'] for x in allrows if x['point']==name and x['task']==t and x['arm']==ARMS[1])-sum(x['success'] for x in allrows if x['point']==name and x['task']==t and x['arm']==ARMS[0]))/20 for t in p['tasks_by_point'][name])/len(p['tasks_by_point'][name])*100)
        independent[str(seed)]=sum(vals)/12
        assert abs(independent[str(seed)]-next(r['delta_seen'] for r in seedrows if r['seed']==seed))<1e-10
    for name,rr in [('trials',allrows),('paired_cells',cellrows),('origin_metrics',originrows),('seed_metrics',seedrows),('task_metrics',taskrows)]:csvwrite(OUT/(name+'.csv'),rr)
    write(OUT/'SUMMARY.json',{'summary':summary,'seeds':seedrows,'tasks':taskrows,'conditional_origin_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist(),'scope':'descriptive conditional on these3development seeds, not population-seed confidence'})
    write(OUT/'DECISION.json',{'decision':decision,'criteria':criteria,'only20step_endpoints':True,'closed_loop_measured':True,'complete_lifelong_training_claim':False})
    write(OUT/'INDEPENDENT_AUDIT.json',{'status':'PASS','trials':len(allrows),'paired_trials':len(pairs),'round_traces_recomputed':len(checks),'independent_seed_deltas':independent,'all_source_hashes':True,'matched_initial_states_and_round_RNG':True})
    text='# Paired closed-loop result\n\n'+decision+'\n\nAll values are success percentage points; QP minus MLR.\n\n| Seed | Seen tasks | Old tasks | New task |\n|---|---:|---:|---:|\n'
    for r in seedrows:text+=f"| {r['seed']} | {r['delta_seen']:.3f} | {r['delta_old']:.3f} | {r['delta_new']:.3f} |\n"
    text+='\nPooled equal-seed results:\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n36 reused origins and3development seeds. All180cells/3600trials included. Only20-step endpoints; not complete lifelong training. Trial counts are not independent seed replications. Per-task costs and conditional descriptive uncertainty are in SUMMARY.json and task_metrics.csv.\n'
    (OUT/'FINAL_REPORT.md').write_text(text)
    print('ANALYSIS_COMPLETE',decision,flush=True)
if __name__=='__main__':run()
