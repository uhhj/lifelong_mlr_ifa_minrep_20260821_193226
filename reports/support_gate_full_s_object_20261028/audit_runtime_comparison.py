"""Compare actual old/new logged time; no profiling or causal bottleneck claim."""
from pathlib import Path
import json,hashlib,statistics
R=Path(__file__).resolve().parent;T=R.parent/'20261026_ccbp_v4_s_object41_42_43'/'final_readonly_20261027_120754'
D=R/'check_readonly_20261028_220233';js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(T/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
old={};sources={str(T/'LOCAL_FINAL_AUDIT.json'):sha(T/'LOCAL_FINAL_AUDIT.json')}
for seed,x in a['seeds'].items():
 ep=T/('seed'+seed)/'light/EPOCHS_S.json';ev=T/('seed'+seed)/'EVALUATION_PROGRESS_AUDIT.json';h=js(ep)
 old[seed]=dict(training_epoch_minutes=x['epoch_seconds']/60,collection_minutes=sum(m['seconds'] for m in x['collection_cost'].values())/60,formal_vector_minutes=sum(js(ev)['vector_seconds'])/60,CPU_minutes=x['final_CPU']['seconds']/60,stage_epoch_means={str(t):statistics.mean(r['seconds'] for r in h if r['stage']==t) for t in range(6,10)})
 sources[str(ep)]=sha(ep);sources[str(ev)]=sha(ev)
v=js(D/'EVIDENCE.json');h=json.loads(v['light']['seed41/EPOCHS_S.json']['text']);t6=[x for x in h if x['stage']==6];t7=[x for x in h if x['stage']==7]
assert len(t6)==100 and len(t7)==60
out=dict(status='READ_ONLY_TIMING_COMPARISON_NOT_PROFILING',observation=v['observed'],old_accepted_full_S=old,new_seed41=dict(task6_minutes=sum(x['seconds'] for x in t6)/60,task7_completed_epochs=60,task7_mean_seconds=statistics.mean(x['seconds'] for x in t7),task7_recent10_mean_seconds=statistics.mean(x['seconds'] for x in t7[-10:])),per_seed_updates=dict(old=105500,new=195300,ratio=195300/105500),known_implementation_costs=['Two candidate trajectories per 10 accepted updates for Task7-9.','Two training probe measurements per gate block plus epoch-end probe.','CPU full-state capture and restore of model/optimizer/scheduler/grads/modes/requires/RNG/step, including equality validation, per block.'],unresolved=['No component-level timing collected; cannot state whether GPU compute, CPU state copying, storage, host load or resource sharing is the primary bottleneck.','Task6 has no gate yet logged time increased, so double-candidate gates cannot explain the whole slowdown.','24-40 hours is a conservative planning extrapolation of current logs, not a necessary algorithmic runtime or a guarantee.'],private_data_downloaded=False,new_model_or_environment_calls=0,sources=dict(historical=sources,current_snapshot_sha256=sha(D/'EVIDENCE.json'),train_source_sha256=sha(R/'seed41/train.py'),runtime_source_sha256=sha(R/'seed41/runtime.py'),probe_source_sha256=sha(R/'seed41/fork_tools.py')))
(R/'RUNTIME_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
