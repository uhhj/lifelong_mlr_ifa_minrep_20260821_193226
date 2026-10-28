"""Local scheduling estimate; stage 8/9 extrapolation is explicitly not measured."""
from pathlib import Path
import json,statistics,hashlib
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261028_214936'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');h=json.loads(v['light']['seed41/EPOCHS_S.json']['text'])
rows=[x for x in h if x['stage']==7];assert len(rows)==48 and a['status']=='PASS_PROGRESS_NOT_FINAL'
recent=[x['seconds'] for x in rows[-10:]];mean=statistics.mean(recent)
collect=json.loads(v['light']['seed41/MEMORY_S_6.json']['text'])['seconds']
stage7=52*mean/60;later=100*(309+362)/227*mean/60;collection=3*collect/60
out=dict(status='SCHEDULING_PROXY_NOT_COMPLETION_CLAIM',observed=a['observation'],seed=41,current_stage=7,complete_current_stage_epochs=48,recent_epoch_seconds=recent,mean_recent=mean,min_recent=min(recent),max_recent=max(recent),mean_stage7=statistics.mean(x['seconds'] for x in rows),measured_current_stage_remaining_compute_minutes=stage7,unmeasured_stages8_9_scaled_compute_minutes=later,unmeasured_remaining_collections_proxy_minutes=collection,whole_seed_proxy_minutes=stage7+later+collection,next_check_minutes=600,assumptions=['Task8/9 scale Task7 recent mean by main batches 309/227 and 362/227; neither later stage has own throughput yet.','Remaining three collections reuse Task6 measured seconds only as a scheduling proxy.'],limitations=['More old tasks and cache windows may make Task8/9 more expensive than batch-linear projection.','Timing fluctuates; excludes checkpoint saving, preparation, admissions, other seeds, formal evaluation and final CPU audit.','No claim that any stage will be complete without a new actual snapshot.'],snapshot_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
