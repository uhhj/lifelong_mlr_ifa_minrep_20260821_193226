"""Read-only timing and q=1 admission inventory from existing snapshots."""
from pathlib import Path
import json,sys,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
js=lambda p:json.loads(p.read_text(encoding='utf8'))
v=js(D/'EVIDENCE.json');audit=js(D/'LOCAL_AUDIT.json');assert audit['status']=='PASS_PROGRESS_NOT_FINAL'
L=v['light'];actual={}
for case in ['Q1_41','Q1_42','Q1_43']:
 def get(n,default=None):return json.loads(L[case+'/'+n]['text']) if case+'/'+n in L else default
 h=get('EPOCHS_S.json',[]);timing={};admissions={}
 for t in (6,7,8,9):
  rr=[x for x in h if x['stage']==t]
  if rr:timing[str(t)]=dict(completed_epochs=len(rr),mean_seconds=statistics.mean(x['seconds'] for x in rr),recent10_mean_seconds=statistics.mean(x['seconds'] for x in rr[-10:]))
  if t>6:
   a=get('ADMISSION_S_%d.json'%t)
   if a:
    assert a['status']=='PASS_OWN_GPU_ADMISSION' and a['q1_actual_memory_uniform_exact'] and a['q1_short_history_exact']
    pr=get('PROBES_%d.json'%t);inp=get('TRAIN_INPUT_S_%d.json'%t)
    for k,raw in pr['empirical_support'].items():
     effective=1. if inp['memory'][k]['count'] else 0.
     assert pr['support'][k]==inp['memory'][k]['effective_support']==effective
     assert inp['memory'][k]['support']==raw
    admissions[str(t)]=dict(status=a['status'],optimizer_calls=a['optimizer_calls'],autograd_calls=a['extra_autograd_calls'],actual_memory_uniform_update_exact=a['q1_actual_memory_uniform_exact'],short_history_fixed_one_exact=a['q1_short_history_exact'],effective_support=pr['support'],empirical_support=pr['empirical_support'],all_replay_state_checks=True)
 actual[case]=dict(timing=timing,admissions=admissions,task6_comparison=get('TASK6_HARDWARE_COMPARISON.json'))
H=R.parent/'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306/EVIDENCE.json';old=js(H)['light'];historical={}
for case in ['L005_42','L020_42','L020_43']:
 h=json.loads(old[case+'/EPOCHS_S.json']['text'])
 historical[case]={str(t):statistics.mean(x['seconds'] for x in h if x['stage']==t) for t in (6,7,8,9)}
reference={str(t):statistics.median(x[str(t)] for x in historical.values()) for t in (6,7,8,9)}
conditional={}
for case,x in actual.items():
 if '7' not in x['timing']:continue
 t=x['timing']['7'];ratio=t['mean_seconds']/reference['7'];remaining=(100-t['completed_epochs'])*t['mean_seconds']
 conditional[case]=dict(current_task7_remaining_minutes=remaining/60,historical_unscaled_stage8_9_minutes=100*(reference['8']+reference['9'])/60,early_task7_speed_ratio=ratio,conditional_remaining_compute_minutes=(remaining+100*(reference['8']+reference['9'])*ratio)/60)
eval_minutes={}
for group in ['L005','L020']:
 total=0
 for seed in (41,42,43):
  events=[json.loads(line) for line in old[f'{group}_{seed}/EVAL_EVENTS.jsonl']['text'].splitlines()]
  total+=max(x['time'] for x in events if x['kind']=='complete')-min(x['time'] for x in events if x['kind']=='start')
 eval_minutes[group]=total/60
observed=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8)))
out=dict(status='CONDITIONAL_PLANNING_NOT_COMPLETION_GUARANTEE',observed=observed.isoformat(),actual=actual,historical_stage_means_seconds=historical,reference_stage_medians_seconds=reference,current_pair_conditional_compute=conditional,historical_full_run_epoch_minutes=sum(reference.values())*100/60,historical_group600_evaluation_minutes=eval_minutes,planning_minutes=dict(current_pair_training_and_collection=[240,330],third_seed_training_and_collection=[240,330],formal600=[60,120],total=[540,780]),planning_completion=[(observed+datetime.timedelta(minutes=m)).isoformat() for m in (540,780)],next_check_minutes=60,sources={str(D/'EVIDENCE.json'):sha(D/'EVIDENCE.json'),str(H):sha(H)},limits=['Current retaining-stage timings only 1 and 3 complete epochs; startup/collection contention and diagnostics influence these samples.','Task8/9 and seed43 not measured in this q1 experiment. Historical same-device S lambda runs are conditional references, not matched-workload controls.','Two initial trainers run in parallel; seed43 begins in a freed slot. Sum of two wave allowances is conservative; do not sum both initial trainers serially.','No guaranteed single-trainer speedup assumed for seed43. Cache contents, gate decisions and GPU resource availability can change times.','Formal evaluation not started; historical 600-episode times are references only.','Final independent CPU acceptance, interpretation and publication are additional.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(out,indent=2,ensure_ascii=False))
