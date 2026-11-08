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
assert a['counts']['Q1_43']['current']['stage']==9 and a['formal_episodes']==0
last=actual['Q1_43']['timing']['9']
remaining=100-last['n']
reference=statistics.mean(actual[c]['timing']['9']['mean_seconds'] for c in ('Q1_41','Q1_42'))
prior=R/'check_readonly_20261108_132454/TIMING_AUDIT.json';old=read(prior)
for p,digest in old['sources'].items():assert sha(Path(p))==digest
obs=datetime.datetime.fromisoformat(a['observation'])
out=dict(status='CONDITIONAL_PLANNING_NOT_FINAL_ACCEPTANCE',observation=a['observation'],actual=actual,
    remaining_compute=dict(epochs=remaining,current_stage_seconds=last['mean_seconds'],
        current_rate_minutes=remaining*last['mean_seconds']/60,
        completed_other_seeds_stage9_seconds=reference,other_seed_reference_minutes=remaining*reference/60),
    planning_minutes=dict(seed43_training_and_collection=[90,120],formal600=[60,120],total=[150,240]),
    planning_completion=[(obs+datetime.timedelta(minutes=m)).isoformat() for m in (150,240)],
    historical_formal600_minutes=old['historical_group600_evaluation_minutes'],next_check_minutes=85,
    sources={str(D/'EVIDENCE.json'):sha(D/'EVIDENCE.json'),str(prior):sha(prior)},
    limits=['Task9 currently has only 8 complete epochs; do not call its throughput stable yet.',
        'Stage8 is fully measured for seed43. Other-seed Task9 reference includes different concurrency/cache conditions.',
        'Training allowance includes remaining epochs, saves and final 10 collection episodes.',
        'Formal600 has not started; historical 59-68 minutes only informs the 60-120 minute allowance.',
        'CPU acceptance, interpretation and publication are extra. Completion requires actual exits.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ('remaining_compute','planning_minutes','planning_completion')},indent=2))
