"""Offline timing review; no remote execution or scientific mutations."""
from pathlib import Path
import datetime, hashlib, json, statistics, sys

R = Path(__file__).resolve().parent
D = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v = read(D/'EVIDENCE.json')
a = read(D/'LOCAL_AUDIT.json')
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL'
prior = R/'check_readonly_20261108_132454/TIMING_AUDIT.json'
reference = read(prior)
for path, digest in reference['sources'].items():
    assert sha(Path(path)) == digest
rates = reference['reference_stage_medians_seconds']
actual = {}
for case in ('Q1_41', 'Q1_42'):
    get = lambda name: json.loads(v['light'][case+'/'+name]['text'])
    epochs = get('EPOCHS_S.json')
    x = [e['seconds'] for e in epochs if e['stage'] == 7]
    assert 40 <= len(x) < 100 and max(e['stage'] for e in epochs) == 7
    adm = get('ADMISSION_S_7.json')
    checks = ['roundtrip_next_step_exact', 'q_one_original_S_exact',
              'lambda_zero_original_C_exact', 'ten_step_C_replay_exact',
              'accepted_and_rejected_full_state_restore', 'measurement_state_exact',
              'probe_all_state_exact', 'q1_actual_memory_uniform_exact',
              'q1_short_history_exact', 'synthetic_gate_accept_reject']
    assert adm['status'] == 'PASS_OWN_GPU_ADMISSION'
    assert all(adm[k] is True for k in checks)
    probe, inp = get('PROBES_7.json'), get('TRAIN_INPUT_S_7.json')
    for task, raw in probe['empirical_support'].items():
        assert inp['memory'][task]['support'] == raw
        assert probe['support'][task] == inp['memory'][task]['effective_support'] == (1. if inp['memory'][task]['count'] else 0.)
    recent = statistics.mean(x[-20:])
    previous = statistics.mean(x[-40:-20])
    ratio = recent/rates['7']
    remaining7 = (100-len(x))*recent/60
    actual[case] = dict(task7_epochs=len(x), mean_seconds=statistics.mean(x),
        recent10_seconds=statistics.mean(x[-10:]), recent20_seconds=recent,
        previous20_seconds=previous, recent20_change_percent=100*(recent/previous-1),
        recent20_cv_percent=100*statistics.pstdev(x[-20:])/recent,
        recent20_range_seconds=[min(x[-20:]), max(x[-20:])],
        nonoverlapping10_means=[statistics.mean(x[i:i+10]) for i in range(0,len(x)-9,10)],
        task7_remaining_compute_minutes=remaining7,
        stage8_9_conditional_minutes=100*(rates['8']+rates['9'])*ratio/60,
        run_remaining_conditional_compute_minutes=remaining7+100*(rates['8']+rates['9'])*ratio/60,
        admission=dict(status=adm['status'], checks={k:adm[k] for k in checks},
            optimizer_calls=adm['optimizer_calls'], autograd_calls=adm['extra_autograd_calls'],
            empirical_support=probe['empirical_support'], effective_support=probe['support']))
assert a['counts']['Q1_43']['epochs'] == 0 and a['formal_episodes'] == 0
observed = datetime.datetime.fromisoformat(a['observation'])
plan = dict(current_parallel_pair_training_collection=[230,280],
            queued_seed43_training_collection=[250,320], formal600=[60,120], total=[540,720])
out = dict(status='CONDITIONAL_PLANNING_NOT_FINAL_ACCEPTANCE', observed=a['observation'],
    actual=actual, reference_stage_seconds=rates,
    historical_formal600_minutes=reference['historical_group600_evaluation_minutes'],
    planning_minutes=plan,
    planning_completion=[(observed+datetime.timedelta(minutes=m)).isoformat() for m in plan['total']],
    next_check_minutes=180,
    sources={str(D/'EVIDENCE.json'):sha(D/'EVIDENCE.json'),str(prior):sha(prior)},
    limits=['Stable refers only to current Task7, with two active trainers. Task8/9 are not measured here.',
        'Future-stage estimates scale historical same-3090 stage means by current recent20 Task7 speed; cache and concurrency may differ.',
        'Current-pair allowance adds preparation, saving, collection, lock waits and variability to conditional pure computation.',
        'Seed43 can start in the first free slot. Two-wave sum is conservative; no assumed single-trainer speedup.',
        'Formal600 has not started. Historical times are references, not current throughput.',
        'CPU acceptance, interpretation and publication are additional; completion is checked by actual exits.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(out,indent=2,ensure_ascii=False))
