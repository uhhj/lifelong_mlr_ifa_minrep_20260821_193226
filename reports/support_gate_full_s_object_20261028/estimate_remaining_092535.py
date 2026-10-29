"""User-requested ETA from audited own-stage timings and explicit references."""
import json,hashlib,statistics,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261029_092535'
T=R.parent/'20261026_ccbp_v4_s_object41_42_43'/'final_readonly_20261027_120754'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
h={s:json.loads(v['light'][f'seed{s}/EPOCHS_S.json']['text']) for s in (41,42)}
assert len(h[41])==400 and len(h[42])==266 and a['counts']['43']['epochs']==0
assert v['light']['seed41/train_S.exit']['text'].strip()=='0'
ref={s:sum(x['seconds'] for x in h[41] if x['stage']==s) for s in range(6,10)}
rows=[x for x in h[42] if x['stage']==8]
coll={s:json.loads(v['light'][f'seed41/MEMORY_S_{s}.json']['text'])['seconds'] for s in range(6,10)}
scenarios={}
for label,subset in [('own_stage8_all66',rows),('own_stage8_recent10',rows[-10:])]:
    sec=statistics.mean(x['seconds'] for x in subset);ratio=sec/(ref[8]/100)
    scenarios[label]=dict(stage8_mean_seconds=sec,stage8_remaining_minutes=34*sec/60,seed42_remaining_training_collection_hours=(34*sec+ref[9]*ratio+coll[8]+coll[9])/3600,seed43_reference_training_collection_hours=(sum(ref.values())*ratio+sum(coll.values()))/3600)
hist_eval=sum(sum(js(T/f'seed{s}'/'EVALUATION_PROGRESS_AUDIT.json')['vector_seconds']) for s in (41,42,43))
now=datetime.datetime.fromisoformat(a['observation'])
out=dict(status='SCENARIO_ETA_NOT_COMPLETION_CLAIM',observation=a['observation'],snapshot_sha256=sha(D/'EVIDENCE.json'),seed41_pure_epoch_hours=sum(ref.values())/3600,seed41_stage_seconds=ref,linear_scenarios=scenarios,historical_600_formal_seconds=hist_eval,planning_hours=dict(seed42_training_collection=[4,5],seed43_training_collection=[8,11],remaining_training_collection_total=[12,16],formal_evaluation=[1,3],training_and_formal_total=[13,19]),training_and_formal_finish_window=[(now+datetime.timedelta(hours=x)).isoformat() for x in (13,19)],final_CPU_report_publication='Additional acceptance/reporting time after runs; not included as completed evidence',next_check_minutes=240,assumptions=['Seed42 Task9 and all seed43 timings are not measured; seed41 same-stage measurements provide reference, scaled by current seed42 Task8 speed.','Planning allowance is wider than linear scenarios for different cache contents, variable shared runtime speed, saving/preparation and admissions; not a confidence interval.','Current formal evaluation has not begun. Historical same-evaluator 600-episode time is a reference, and 1-3h is a planning allowance, not a current measurement.','No new infrastructure interruption or engineering failure assumed.','Final independent CPU acceptance and scientific interpretation remain pending even when training and evaluation finish.'],sources=dict(historical_final_audit_sha256=sha(T/'LOCAL_FINAL_AUDIT.json'),historical_evaluation_audit_sha256={str(s):sha(T/f'seed{s}'/'EVALUATION_PROGRESS_AUDIT.json') for s in (41,42,43)}))
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
