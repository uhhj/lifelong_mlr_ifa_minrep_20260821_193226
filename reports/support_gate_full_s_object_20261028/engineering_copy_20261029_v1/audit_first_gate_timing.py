"""Read-only actual first gated-stage admission and conditional scheduling estimate."""
from pathlib import Path
import json,hashlib,statistics,sys
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'));a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));L=v['light']
get=lambda n:json.loads(L[n]['text'])
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
cap=get('engineering_copy_20261029_v1/SEED43_CAPTURE_EQ.json')
assert cap['checked']['0']['all_state_exact'] and cap['checked']['15700']['all_state_exact']
adm=get('seed43/ADMISSION_S_7.json');assert adm['status']=='PASS_OWN_GPU_ADMISSION' and adm['optimizer_calls']==29 and adm['extra_autograd_calls']==11
assert all(adm[k] for k in ['roundtrip_next_step_exact','q_one_original_S_exact','lambda_zero_original_C_exact','ten_step_C_replay_exact','accepted_and_rejected_full_state_restore','measurement_state_exact','probe_all_state_exact'])
mem=get('seed43/MEMORY_S_6.json');assert mem['training_attempts']==10 and mem['collection_successes']==6 and mem['count']==76
assert get('seed43/PROBES_7.json')['support']=={'6':.6}
hist={s:get('seed%d/EPOCHS_S.json'%s) for s in [41,42,43]}
def mean(seed,stage,recent=False):
 rows=[r['seconds'] for r in hist[seed] if r['stage']==stage]
 return statistics.mean(rows[-10:] if recent else rows)
n=sum(x['stage']==7 for x in hist[43]);assert n>=10
current_minutes=(100-n)*mean(43,7,True)/60
ratio=mean(43,7)/mean(42,7)
reference_future=sum(mean(42,s)*100 for s in [8,9])/60
collection_reference=sum(get('seed42/MEMORY_S_%d.json'%s)['seconds'] for s in [7,8,9])/60
timing=dict(status='CONDITIONAL_SCHEDULING_ESTIMATE_NOT_SPEEDUP_ACCEPTANCE',observation=a['observation'],evidence_sha256=a['snapshot_sha256'],current=dict(seed=43,stage=7,completed_epochs=n,all_mean_seconds=mean(43,7),recent10_mean_seconds=mean(43,7,True),remaining_stage_compute_minutes=current_minutes),references={str(s):{str(t):mean(s,t) for t in [7,8,9]} for s in [41,42]},seed43_to_seed42_stage7_mean_ratio=ratio,unmeasured_stage8_9_reference_compute_minutes=reference_future,remaining_collection_reference_minutes=collection_reference,remaining_seed_training_collection_linear_minutes=[current_minutes+reference_future*ratio+collection_reference,current_minutes+reference_future+collection_reference],planning_training_hours=[5,7],planning_all_training_and_formal_hours=[6,10],next_check_minutes=270,assumptions=['Only current seed43 stage7 has measured optimized gated throughput; stages8/9 remain unmeasured.','Lower linear scenario scales seed42 stages8/9 by current stage7 speed ratio; upper scenario makes no such credit.','Later caches, history-length grouping, preparation/checkpoint overhead and resource fluctuation may change speed.','Cross-seed timing ratios do not isolate clone optimization: data masks, acquired caches/support and machine conditions differ.','Formal 600 episodes are unstarted; 1-3h remains a historical planning allowance, not current measured evaluation throughput.','Final independent CPU acceptance/report/publication are additional.','No scientific benefit is inferred from gate acceptance or training probes.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
proof=dict(status='PASS_ACTUAL_OPTIMIZED_STAGE7_NOT_FINAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],own_admission=True,stage6_and_7_actual_full_state_copy_exact=True,own_stage7_optimizer_calls=29,own_stage7_extra_autograd=11,task6_collection_attempts=10,task6_successful_trajectories=6,task6_cache_windows=76,q6=.6,future_stage8_9_admission='PENDING_NOT_INHERITED',new_optimizer_calls_from_this_audit=0,new_environments=0)
(D/'OPTIMIZED_STAGE7_LOCAL_AUDIT.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(proof=proof,timing=timing),indent=2))
