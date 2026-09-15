"""User-requested staging: finish seed45, then pause without starting46/47."""
import os,time,traceback,statistics,json
from pathlib import Path
from .common import *
from .queue import run_child
def active(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'
def summarize():
    import numpy as np
    from .analyze import csvwrite
    p=read(OUT/'PROTOCOL_FROZEN.json');names=[n for n in p['points'] if n.startswith('s45_')];cellrows=[];originrows=[];alltrials=[]
    assert len(names)==12
    for name in names:
        cur=int(name.split('_')[1][1:]);tasks=p['tasks_by_point'][name];v={a:{} for a in ARMS}
        for task in tasks:
            rounds={};trials={};meta={}
            for arm in ARMS:
                dest=OUT/'evaluation'/name/f'task{task}'/arm;meta[arm]=read(dest/'COMPLETE.json')
                for f,h in meta[arm]['files'].items():assert sha(dest/f)==h
                rr=read(dest/'ROUNDS.json');tt=read(dest/'TRIALS.json');assert len(tt)==20 and [x['trial'] for x in tt]==list(range(20))
                alltrials+=tt;rounds[arm]=rr;trials[arm]=tt
                derived=[]
                for r in range(2):
                    with np.load(dest/f'round{r}.npz') as z:dd=trial_summary(z['done'],z['steps'])
                    assert [x['success'] for x in dd]==rr[r]['success'];assert [x['first_success_step'] for x in dd]==rr[r]['first']
                    derived.extend(x['success'] for x in dd)
                assert derived==[bool(x['success']) for x in tt]
                v[arm][task]=sum(derived)*5.
            for r in range(2):
                for k in ['init_sha256','initial_obs_sha256','policy_seed','policy_rng_before']:assert rounds[ARMS[0]][r][k]==rounds[ARMS[1]][r][k],(name,task,r,k)
                if meta[ARMS[0]]['checkpoint_model_hash']==meta[ARMS[1]]['checkpoint_model_hash']:
                    for k in ['action_sha256','done_sha256','success','first']:assert rounds[ARMS[0]][r][k]==rounds[ARMS[1]][r][k]
            wins=sum(y['success']>x['success'] for x,y in zip(trials[ARMS[0]],trials[ARMS[1]]));losses=sum(y['success']<x['success'] for x,y in zip(trials[ARMS[0]],trials[ARMS[1]]))
            cellrows.append(dict(point=name,current_task=cur,eval_task=task,MLR=v[ARMS[0]][task],TASK_GRAD_QP=v[ARMS[1]][task],delta_pp=v[ARMS[1]][task]-v[ARMS[0]][task],paired_wins=wins,paired_losses=losses))
        row=dict(point=name,current_task=cur)
        for arm in ARMS:
            row[arm+'_seen']=statistics.mean(v[arm].values());row[arm+'_old']=statistics.mean(x for t,x in v[arm].items() if t<cur);row[arm+'_new']=v[arm][cur];row[arm+'_worst_old']=min(x for t,x in v[arm].items() if t<cur)
        for m in ['seen','old','new','worst_old']:row['delta_'+m]=row[ARMS[1]+'_'+m]-row[ARMS[0]+'_'+m]
        originrows.append(row)
    assert len(alltrials)==1200 and len(cellrows)==30
    summary={k:statistics.mean(r[k] for r in originrows) for k in originrows[0] if k not in ['point','current_task']}
    # Independent count-based oracle, not the above success-rate table.
    check=statistics.mean([statistics.mean([(sum(r['success'] for r in alltrials if r['point']==n and r['task']==t and r['arm']==ARMS[1])-sum(r['success'] for r in alltrials if r['point']==n and r['task']==t and r['arm']==ARMS[0]))*5 for t in p['tasks_by_point'][n]]) for n in names])
    assert abs(check-summary['delta_seen'])<1e-10
    taskrows=[dict(task=t,**{a:statistics.mean(r[a] for r in cellrows if r['eval_task']==t) for a in ARMS},delta_pp=statistics.mean(r['delta_pp'] for r in cellrows if r['eval_task']==t)) for t in [6,7,8]]
    dest=OUT/'seed45_pilot';dest.mkdir(exist_ok=True)
    for fn,rr in [('trials',alltrials),('paired_cells',cellrows),('origins',originrows),('tasks',taskrows)]:csvwrite(dest/(fn+'.csv'),rr)
    write(dest/'SUMMARY.json',dict(seed=45,summary=summary,tasks=taskrows,primary_trials=1200,origins=12,scope='Single development seed only; three-seed decision NOT EVALUATED; no automatic progression based on this result'))
    write(dest/'INDEPENDENT_AUDIT.json',dict(status='PASS',trial_count=1200,paired_inputs=True,all_raw_done_recomputed=True,independent_primary_delta=check,identical_endpoint_null_pairs_checked=True))
    write(dest/'DECISION.json',dict(status='SINGLE_SEED_EXPLORATORY_ONLY',three_seed_confirmation='NOT_EVALUATED',remaining_seeds=[46,47],queue_action='PAUSE_AS_REQUESTED_REGARDLESS_OF_RESULT'))
    (dest/'REPORT.md').write_text('# Seed45 closed-loop pilot\n\nAll12origins,60cells,1200trials. QP minus MLR success percentage points.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nOnly one development seed. Original three-seed decision not evaluated. Per-task and per-origin results accompany this report. Seeds46/47 deferred at user request, regardless of outcome.\n')
def run():
    verify();start=time.time();scope=read(OUT/'EXECUTION_SCOPE.json');assert scope['active_seeds']==[45]
    write(OUT/'SEED45_QUEUE_STARTED.json',dict(time=start,pid=os.getpid(),active_seeds=[45],deferred_seeds=[46,47],adopted_preflight_child=scope['adopted_child_pid']))
    write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage='adopt_hashfix_trace_off',pid=os.getpid(),child_pid=scope['adopted_child_pid'],started=start,formal_completed_cells=0,active_seeds=[45]))
    while active(scope['adopted_child_pid']):
        assert time.time()-start<3600,'adopted preflight watchdog';time.sleep(5)
    assert (OUT/'engineering/trace_off_hashfix/COMPLETE.json').exists(),'adopted preflight did not complete'
    run_child('hashfix_trace_on',['-m','paper_qp_rollout_v1.evaluate','0','trace_on_hashfix'],start)
    off=read(OUT/'engineering/trace_off_hashfix/ROUNDS.json');on=read(OUT/'engineering/trace_on_hashfix/ROUNDS.json');assert len(off)==len(on)==2
    for x,y in zip(off,on):
        yy=dict(y);yy.pop('trace_file_sha256');assert x==yy,('content-hash parity failed',x,y)
    for tag,rr in [('trace_off',off),('trace_on',on)]:
        for x,y in zip(read(OUT/'engineering'/tag/'ROUNDS.json'),rr):
            for k in ['init_sha256','policy_rng_before','policy_rng_after','count','success','first','action_sha256','done_sha256']:assert x[k]==y[k],('technical hash fix affected behavior',tag,k)
    write(OUT/'PREFLIGHT_COMPLETE.json',dict(status='PASS',recovery_origins=6,full_witnesses=36,content_hash_and_full_rollout_parity=True,engineering_trials_excluded=80))
    (OUT/'PRECHECK_REPORT.md').write_text('PASS: original6origin full-state recovery; repeated20trial rollout has identical observation contents,actions,dones,RNG. Original faulty pointer-hash evidence retained;80engineering trials excluded. User stages main evaluation to seed45 only,12origins/60cells/1200trials. Seeds46/47 deferred.\n')
    run_child('recover_seed45',['-m','paper_qp_rollout_v1.recover','seed45'],start)
    write(OUT/'FORMAL_STARTED.json',dict(time=time.time(),pid=os.getpid(),active_seeds=[45],cells=60,trials=1200,full_original_protocol_cells=180))
    for i,c in enumerate(read(OUT/'PROTOCOL_FROZEN.json')['cells']):
        if c['point'].startswith('s45_'):run_child('cell_'+str(i).zfill(3),['-m','paper_qp_rollout_v1.evaluate',str(i)],start)
    summarize()
    write(OUT/'QUEUE_STATUS.json',dict(status='PAUSED_AFTER_SEED45',time=time.time(),seconds=time.time()-start,completed_seeds=[45],deferred_seeds=[46,47],formal_completed_cells=60,trials=1200,reason='User-requested staging fixed before main outcomes; pause regardless of scores'))
if __name__=='__main__':
    try:run()
    except BaseException:write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',time=time.time(),traceback=traceback.format_exc()));raise
