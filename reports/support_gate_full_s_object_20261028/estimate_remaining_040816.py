"""Read-only timing scenarios; no training or environment operations."""
import hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'check_readonly_20261029_040816'
v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
h=json.loads(v['light']['seed41/EPOCHS_S.json']['text'])
stages={}
for s in range(6,10):
    rows=[x for x in h if x['stage']==s]
    stages[str(s)]=dict(epochs=len(rows),pure_epoch_seconds=sum(x['seconds'] for x in rows),mean_seconds=statistics.mean(x['seconds'] for x in rows),recent10_seconds=statistics.mean(x['seconds'] for x in rows[-10:]),recent10_range=[min(x['seconds'] for x in rows[-10:]),max(x['seconds'] for x in rows[-10:])])
assert [stages[str(s)]['epochs'] for s in range(6,10)]==[100,100,100,66]
collections={str(s):json.loads(v['light'][f'seed41/MEMORY_S_{s}.json']['text'])['seconds'] for s in (6,7,8)}
out=dict(status='SCENARIO_ETA_NOT_COMPLETION_CLAIM',observation=a['observation'],snapshot_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),stages=stages,collection_seconds=collections,stage9_remaining_pure_compute_minutes={label:34*stages['9'][key]/60 for label,key in [('all66','mean_seconds'),('recent10','recent10_seconds')]},next_check_minutes=150,limits=['Current Task9 own timings replace previous Task7 extrapolation for seed41.','Remaining 34 complete epochs include the currently partial epoch; no partial-epoch cost counted as complete.','One final ten-episode collection, checkpoint saving and preparation remain; current collection time is unknown.','Seeds42/43 and formal evaluation have not started; their throughput is unmeasured. Prior total 24-40h estimate is historical, not a refreshed completion promise.','Recent Task9 times vary materially; no component profiling or claim of bottleneck cause.','150-minute check targets the current seed completion vicinity, not a fixed polling cadence.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
