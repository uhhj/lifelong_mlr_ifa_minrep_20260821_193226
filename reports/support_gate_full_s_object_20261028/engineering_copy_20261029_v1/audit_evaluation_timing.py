"""Estimate remaining evaluation from completed vector events; no environment calls."""
from pathlib import Path
import datetime, json, statistics, sys
D=Path(sys.argv[1])
v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
L=v['light']; get=lambda n:json.loads(L[n]['text'])
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert a['epochs']==1200 and a['training_collection_completed_episodes']==120
assert a['accepted_updates']==316500 and a['physical_training_optimizer_calls']==585900
assert all(L['seed%d/train_S.exit'%s]['text'].strip()=='0' for s in [41,42,43])
events=[]
for s in [41,42,43]:
    x=L.get('seed%d/EVAL_EVENTS.jsonl'%s)
    if x:
        events.extend(dict(json.loads(line), seed=s) for line in x['text'].splitlines() if line.strip())
complete=[r for r in events if r['kind']=='complete']
assert complete and 0<a['formal_episodes']<600
times=[r['seconds'] for r in complete]
remaining_vectors=60-len(complete)
hist={s:get('seed%d/EPOCHS_S.json'%s) for s in [41,42,43]}
hours={s:sum(r['seconds'] for r in hist[s])/3600 for s in hist}
obs=datetime.datetime.fromisoformat(a['observation'])
out=dict(status='PARTIAL_EVALUATION_MEASURED_TIMING_NOT_FINAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],outcome_episodes=a['formal_episodes'],complete_vector_events=len(complete),complete_event_episodes=len(complete)*10,mean_vector_seconds=statistics.mean(times),range_vector_seconds=[min(times),max(times)],remaining_vectors_including_inflight=remaining_vectors,remaining_minutes_mean=remaining_vectors*statistics.mean(times)/60,remaining_minutes_observed_max=remaining_vectors*max(times)/60,planning_remaining_formal_minutes=[60,90],planning_finish_window=[(obs+datetime.timedelta(minutes=m)).isoformat() for m in [60,90]],next_check_minutes=70,completed_epoch_compute_hours=hours,seed43_vs_seed42_epoch_time_reduction_fraction=1-hours[43]/hours[42],limitations=['Snapshot reads outcomes and events sequentially; an outcome vector may precede its complete event. Remaining estimate conservatively includes that in-flight vector.', 'Only completed vector event durations are used, not differences between episode timestamps.', 'Later tasks, episode lengths and resource conditions may differ; estimate is conditional.', 'Epoch compute excludes saves, preparation and collection; cross-seed speed differences do not identify causal optimization gain.', 'Independent final CPU acceptance, analysis and publication remain additional.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
