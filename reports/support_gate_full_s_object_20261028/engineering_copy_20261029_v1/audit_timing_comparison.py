"""Matched-prefix observed timing comparison, not a causal optimization benchmark."""
from pathlib import Path
import json,statistics,sys,datetime,hashlib
D=Path(sys.argv[1]);N=D.parent
v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'));a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));L=v['light'];get=lambda n:json.loads(L[n]['text'])
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
h={s:get('seed%d/EPOCHS_S.json'%s) for s in [41,42,43]};z=[x['seconds'] for x in h[43] if x['stage']==7];n=len(z);assert 10<=n<=100
comparison={}
for seed in [41,42]:
 b=[x['seconds'] for x in h[seed] if x['stage']==7][:n];assert len(b)==n
 bm,zm=statistics.mean(b),statistics.mean(z);bd,zd=statistics.median(b),statistics.median(z)
 comparison[str(seed)]=dict(matched_first_epochs=n,original_mean_seconds=bm,optimized_mean_seconds=zm,mean_time_reduction_fraction=1-zm/bm,observed_mean_throughput_ratio=bm/zm,original_median_seconds=bd,optimized_median_seconds=zd,median_time_reduction_fraction=1-zd/bd,original_min_max=[min(b),max(b)],optimized_min_max=[min(z),max(z)])
current_minutes=(100-n)*statistics.mean(z[-10:])/60
future_reference=sum(statistics.mean([r['seconds'] for r in h[42] if r['stage']==s])*100 for s in [8,9])/60
ratio=statistics.mean(z)/statistics.mean([r['seconds'] for r in h[42] if r['stage']==7][:n])
collection_reference=sum(get('seed42/MEMORY_S_%d.json'%s)['seconds'] for s in [7,8,9])/60
obs=datetime.datetime.fromisoformat(a['observation']);historical=N/'check_readonly_20261029_092535/TIMING_AUDIT.json';old=json.loads(historical.read_text(encoding='utf8'))
cap=get('engineering_copy_20261029_v1/SEED43_CAPTURE_EQ.json');assert cap['checked']['15700']['all_state_exact']
out=dict(status='OBSERVED_MATCHED_PREFIX_COMPARISON_AND_CONDITIONAL_ETA',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],stage7_completed=n,current_recent10_seconds=statistics.mean(z[-10:]),comparisons=comparison,remaining_current_stage_compute_minutes=current_minutes,unmeasured_stage8_9_seed42_reference_minutes=future_reference,remaining_training_collection_linear_minutes=[current_minutes+future_reference*ratio+collection_reference,current_minutes+future_reference+collection_reference],planning_hours=dict(training_collection=[4,6],all_three_seed_formal=[1,3],total_training_and_formal=[5,9]),training_finish_window=[(obs+datetime.timedelta(hours=x)).isoformat() for x in [4,6]],training_and_all_formal_finish_window=[(obs+datetime.timedelta(hours=x)).isoformat() for x in [5,9]],historical_formal_seconds=old['historical_600_formal_seconds'],historical_timing_audit_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),next_check_minutes=210,limitations=['Times measure epoch computation, with epoch checkpoint saving and stage preparation/collection outside that timer.','Matched epoch counts are cross-seed observations, not randomized same-state timing tests; data masks, own caches, grouped retention forwards and resource conditions differ.','A mean time reduction is not identical to a throughput increase; both are reported separately.','Original seed42 contains large slow outliers; median comparison reduces their influence but is still not a causal estimate.','No isolated end-to-end causal speedup has been measured; earlier copy-primitive timing and conditional Amdahl estimate remain separate.','Seed43 Task8/9 and current 600 formal episodes are unmeasured; scenarios are scheduling allowances, not guarantees.','Final independent CPU acceptance, analysis and publication are additional; no training or environment was executed by this audit.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
