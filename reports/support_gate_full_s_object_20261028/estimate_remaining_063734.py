"""Measured seed41 reference and seed42 scheduling scenarios, not acceptance."""
import json,hashlib,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261029_063734'
v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['light']['seed41/train_S.exit']['text'].strip()=='0'
def js(n):return json.loads(v['light'][n]['text'])
h41=js('seed41/EPOCHS_S.json');h42=js('seed42/EPOCHS_S.json')
assert len(h41)==400 and len(h42)==137
stages={str(s):sum(x['seconds'] for x in h41 if x['stage']==s) for s in range(6,10)}
collection={str(s):js(f'seed41/MEMORY_S_{s}.json')['seconds'] for s in range(6,10)}
rows=[x for x in h42 if x['stage']==7]
scenarios={}
for name,rows_used in [('own_all37',rows),('own_recent10',rows[-10:])]:
    mean=statistics.mean(x['seconds'] for x in rows_used)
    ratio=mean/(stages['7']/100)
    scenarios[name]=dict(stage7_mean_seconds=mean,stage7_remaining_minutes=63*mean/60,seed42_remaining_training_collection_hours=(63*mean+ratio*(stages['8']+stages['9'])+sum(collection[str(s)] for s in (7,8,9)))/3600,reference_scaling_ratio=ratio)
out=dict(status='SCENARIO_ETA_NOT_COMPLETION_CLAIM',observation=a['observation'],snapshot_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),seed41_stage_epoch_seconds=stages,seed41_collection_seconds=collection,seed41_pure_epoch_hours=sum(stages.values())/3600,seed42_scenarios=scenarios,next_check_minutes=480,limits=['Seed41 training exit0 and 40 collections are complete; formal evaluation and independent final CPU acceptance remain pending.','Seed42 Task8/9 have not run. Their scheduling reference is seed41 same-stage time scaled by the ratio of seed42 own Task7 time to seed41 Task7 mean.','Different seed caches, resource variation, checkpoint saving, preparation and admissions can change future speed; this is not a confidence interval or promise.','Seed43 and current formal evaluation throughput are still unmeasured; do not renew the earlier total estimate as measured fact.','Next check is near the earlier estimated current-seed completion, not a fixed short polling cadence.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
