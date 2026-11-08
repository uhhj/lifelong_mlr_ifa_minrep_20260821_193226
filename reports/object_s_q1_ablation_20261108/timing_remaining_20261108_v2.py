"""Offline audit of completed stages and remaining single-seed work."""
from pathlib import Path
import datetime, hashlib, json, statistics, sys
R=Path(__file__).resolve().parent
D=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(D/'EVIDENCE.json');a=read(D/'LOCAL_AUDIT.json');L=v['light']
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
checks=['roundtrip_next_step_exact','q_one_original_S_exact','lambda_zero_original_C_exact',
        'ten_step_C_replay_exact','accepted_and_rejected_full_state_restore',
        'measurement_state_exact','probe_all_state_exact','q1_actual_memory_uniform_exact',
        'q1_short_history_exact','synthetic_gate_accept_reject']
actual={}
for c in ('Q1_41','Q1_42','Q1_43'):
    get=lambda name:json.loads(L[c+'/'+name]['text'])
    h=get('EPOCHS_S.json');timings={};admissions={}
    for t in (6,7,8,9):
        x=[e['seconds'] for e in h if e['stage']==t]
        if x:
            z=dict(n=len(x),mean_seconds=statistics.mean(x),recent10_seconds=statistics.mean(x[-10:]),
                   recent20_seconds=statistics.mean(x[-20:]),recent20_cv_percent=100*statistics.pstdev(x[-20:])/statistics.mean(x[-20:]))
            if len(x)>=40:
                z['previous20_seconds']=statistics.mean(x[-40:-20])
                z['recent20_change_percent']=100*(z['recent20_seconds']/z['previous20_seconds']-1)
            timings[str(t)]=z
        if t>6 and c+'/ADMISSION_S_%d.json'%t in L:
            adm=get('ADMISSION_S_%d.json'%t)
            assert adm['status']=='PASS_OWN_GPU_ADMISSION' and all(adm[k] is True for k in checks)
            pr=get('PROBES_%d.json'%t);inp=get('TRAIN_INPUT_S_%d.json'%t)
            for task,raw in pr['empirical_support'].items():
                assert inp['memory'][task]['support']==raw
                assert pr['support'][task]==inp['memory'][task]['effective_support']==(1. if inp['memory'][task]['count'] else 0.)
            admissions[str(t)]=dict(status=adm['status'],checks={k:adm[k] for k in checks},
                optimizer_calls=adm['optimizer_calls'],autograd_calls=adm['extra_autograd_calls'],
                empirical_support=pr['empirical_support'],effective_support=pr['support'])
    actual[c]=dict(timing=timings,admissions=admissions)
assert all(a['counts'][c]['epochs']==400 and L[c+'/train_S.exit']['text'].strip()=='0' for c in ('Q1_41','Q1_42'))
assert a['counts']['Q1_43']['current']['stage']==7 and a['formal_episodes']==0
ref={str(t):statistics.mean(actual[c]['timing'][str(t)]['mean_seconds'] for c in ('Q1_41','Q1_42')) for t in (6,7,8,9)}
last=actual['Q1_43']['timing']['7'];ratio=last['recent20_seconds']/ref['7']
remain7=(100-last['n'])*last['recent20_seconds']/60
unscaled=100*(ref['8']+ref['9'])/60
prior=R/'check_readonly_20261108_132454/TIMING_AUDIT.json';old=read(prior)
for p,digest in old['sources'].items():assert sha(Path(p))==digest
events=[json.loads(s) for s in L['CONTROLLER_EVENTS.jsonl']['text'].splitlines()]
tz=datetime.timezone(datetime.timedelta(hours=8));obs=datetime.datetime.fromisoformat(a['observation'])
out=dict(status='CONDITIONAL_PLANNING_NOT_FINAL_ACCEPTANCE',observation=a['observation'],actual=actual,
    controller_events=[dict(kind=e['kind'],case=e['case'],time=datetime.datetime.fromtimestamp(e['time'],tz).isoformat(),returncode=e.get('returncode')) for e in events],
    current_experiment_completed_stage_reference_seconds=ref,
    remaining_compute=dict(task7_minutes=remain7,stage8_9_unscaled_minutes=unscaled,
        task7_speed_ratio=ratio,total_scaled_minutes=remain7+unscaled*ratio,total_unscaled_minutes=remain7+unscaled),
    planning_minutes=dict(seed43_training_and_collection=[210,260],formal600=[60,120],total=[270,380]),
    planning_completion=[(obs+datetime.timedelta(minutes=m)).isoformat() for m in (270,380)],
    historical_formal600_minutes=old['historical_group600_evaluation_minutes'],next_check_minutes=180,
    sources={str(D/'EVIDENCE.json'):sha(D/'EVIDENCE.json'),str(prior):sha(prior)},
    limits=['Stage8/9 references now come from completed q1 seed41/42, not lambda experiments.',
        'Seed43 stage8/9 are unmeasured; scaling by its current Task7 speed is conditional, not guaranteed.',
        'Remaining training allowance includes stage admission, preparation, saves and 30 collection episodes.',
        'Formal600 is not started; historical same-device times are references only.',
        'Independent CPU acceptance, interpretation and publication are extra. No science was rerun.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ('controller_events','remaining_compute','planning_minutes','planning_completion')},indent=2))
