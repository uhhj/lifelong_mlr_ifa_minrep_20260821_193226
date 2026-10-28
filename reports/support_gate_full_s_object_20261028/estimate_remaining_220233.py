"""Scheduling scenarios for a user-requested ETA; no scientific operations."""
import json,hashlib,statistics,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261028_220233'
T=R.parent/'20261026_ccbp_v4_s_object41_42_43'/'final_readonly_20261027_120754'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json')
h=json.loads(v['light']['seed41/EPOCHS_S.json']['text']);rows=[x for x in h if x['stage']==7]
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and len(rows)==60 and a['counts']['42']['epochs']==a['counts']['43']['epochs']==0
fast=statistics.mean(x['seconds'] for x in rows[-10:]);long=statistics.mean(x['seconds'] for x in rows)
t6=sum(x['seconds'] for x in h if x['stage']==6)
c=json.loads(v['light']['seed41/MEMORY_S_6.json']['text'])['seconds']
hist_eval=sum(sum(js(T/f'seed{s}'/'EVALUATION_PROGRESS_AUDIT.json')['vector_seconds']) for s in [41,42,43])
hist_CPU=sum(x['final_CPU']['seconds'] for x in js(T/'LOCAL_FINAL_AUDIT.json')['seeds'].values())
scenarios={}
for label,sec in [('recent10',fast),('all60',long)]:
 seed_gate=(40+100*(309+362)/227)*sec
 all_gate=(40+100*(309+362)/227+2*100*(227+309+362)/227)*sec
 scenarios[label]=dict(stage7_remaining_minutes=40*sec/60,seed41_training_plus_collection_hours=(seed_gate+3*c)/3600,all_seeds_training_plus_collection_hours=(all_gate+2*t6+11*c)/3600)
now=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8)))
out=dict(status='SCENARIO_ETA_NOT_COMPLETION_CLAIM',observation=now.isoformat(),recent10_epoch_seconds=fast,all60_epoch_seconds=long,stage7_recent_range=[min(x['seconds'] for x in rows[-10:]),max(x['seconds'] for x in rows[-10:])],own_stage6_training_seconds=t6,own_stage6_collection_seconds=c,historical_600_formal_vector_seconds=hist_eval,historical_3_seed_CPU_seconds=hist_CPU,linear_scenarios=scenarios,user_planning_range_hours=dict(seed41=[6,10],all_training_formal_and_acceptance=[24,40]),all_finish_window=[(now+datetime.timedelta(hours=n)).isoformat() for n in [24,40]],seed41_finish_window=[(now+datetime.timedelta(hours=n)).isoformat() for n in [6,10]],next_check_minutes=360,assumptions=['Task8/9 and seed42/43 gate throughput not measured; scaled by main-batch count from Task7.','Planning range adds allowance for more old-task/cache probe work, fluctuation, checkpoint/preparation/admission time; not a statistical confidence interval.','Current formal evaluation is not started. Historical identical evaluator timings are a reference only; allow roughly 2–4 hours for formal evaluation, CPU audit, reporting/publication under current variability.','No new outage, engineering failure, or changed resource allocation assumed.'],limits=['Do not claim future stages have run, or use elapsed clock time as completion evidence.','Task7 current remaining time is not seed41 full-sequence time or three-seed total time.','Canceled mechanism results/costs remain separate and are never resumed.'],sources=dict(snapshot_sha256=sha(D/'EVIDENCE.json'),historical_final_audit_sha256=sha(T/'LOCAL_FINAL_AUDIT.json'),historical_evaluation_audit_sha256={str(s):sha(T/f'seed{s}'/'EVALUATION_PROGRESS_AUDIT.json') for s in [41,42,43]}))
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
