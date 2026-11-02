"""Independently account returned diagnostic evidence; never start a GPU job."""
import json,hashlib,statistics,sys
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
js=lambda p:json.loads(p.read_text(encoding='utf8'))
inputs=[('probe',R/'capacity_readonly_20261102_160106','PASS_STAGE9_PARALLEL_EXACT_FEASIBILITY',2904),('timing',R/'capacity_readonly_20261102_160738','PASS_STAGE9_PARALLEL_THROUGHPUT_EXACT_ENDSTATE',2896),('full',Path(sys.argv[1]),'PASS_REMAINING_SEVEN_STAGE_PARALLEL_EXACT',12784)]
results={};grid=set();cost=0
for label,d,status,expected in inputs:
 v=js(d/'EVIDENCE.json');f=v['files'];z=f['RESULT.json']
 assert z['status']==status and z['optimizer_started']==z['optimizer_completed']==expected
 assert z['environments']==z['production_epochs']==z['extra_autograd']==z['production_signals']==0 and not z['production_writes']
 assert not v['driver'] or v['driver']['state']=='Z'
 assert all(not p or p['state']=='Z' for p in v['workers'])
 if label=='full':
  assert not v['continuation_driver'] or v['continuation_driver']['state']=='Z'
  assert 'CONTINUATION_FAILURE.json' not in f and 'Insufficient diagnostic headroom' in f['FAILURE.json']['error']
  assert len(f['FAILURE.json']['diagnostic_children'])==2 and all(x['returncode']==0 for x in f['FAILURE.json']['diagnostic_children'])
  assert z['continuation_current_optimizer_calls']==z['retained_serial_optimizer_calls']==6392
  assert v['continuation_source_sha256']==z['continuation_source_sha256']==hashlib.sha256((E/'full_continue_v2.py').read_text(encoding='utf8').encode()).hexdigest()
  assert z['resolved_guard_failure_sha256']==f['CONTINUATION_DISPATCH.json']['prior_guard_failure_sha256']
 else:assert not any(k.endswith('FAILURE.json') for k in f)
 hashes=v.get('source_hashes',f['SOURCE_MANIFEST.json'])
 assert f['SOURCE_MANIFEST.json']==hashes==f['DISPATCH.json']['files']
 names={'probe':('worker.py','driver.py'),'timing':('perf_worker.py','perf_driver.py'),'full':('full_worker.py','full_driver.py')}[label]
 for remote,local in zip(['worker.py','driver.py'],names):
  key=f['DISPATCH.json']['directory']+'/'+remote
  assert hashes[key]==hashlib.sha256((E/local).read_text(encoding='utf8').encode()).hexdigest()
 workers=[x for k,x in f.items() if k.endswith('/RESULT.json')];assert len(workers)==4
 assert sum(w['optimizer_completed'] for w in workers)==expected and all(w['optimizer_started']==w['optimizer_completed'] for w in workers)
 for report in z['reports']:
  if label!='timing':
   assert report['all_per_step_scalars_branch_state_digests_probes_decisions_and_next_step_exact'];grid.add((report['lambda_R'],report['stage']))
  else:assert report['per_step_scalars_probes_decisions_initial_final_state_exact']
  s=report['serial_two_worker_seconds'];p=report['parallel_two_worker_seconds'];assert report['aggregate_throughput_ratio']==s/p and report['wall_reduction_fraction']==1-p/s
 result=dict(status=z['status'],optimizer_calls=expected,environments=0,extra_autograd=0,source=str(d),evidence_sha256=sha(d/'EVIDENCE.json'),reports=z['reports'],all_diagnostic_processes_exited=True,source_rehashed_in_status_snapshot='source_hashes' in v,all_remote_source_hashes_must_be_rechecked_by_production_dispatch=True)
 if label=='timing':
  windows={}
  for mode in ['serial42','serial43','parallel']:
   ds=[x for k,x in f.items() if k.startswith(mode+'_') and k.endswith('0.05_9_DONE.json')]
   lo=min(x['started'] for x in ds);hi=max(x['ended'] for x in ds)
   xs=[list(map(float,x['gpu'].split(','))) for x in f['METRICS.json'] if lo<=x['time']<=hi and x['exit']==0]
   windows[mode]=dict(samples=len(xs),gpu_mean=statistics.mean(x[0] for x in xs),gpu_min=min(x[0] for x in xs),gpu_max=max(x[0] for x in xs),memory_peak_MiB=max(x[1] for x in xs),temperature_peak_C=max(x[2] for x in xs),power_mean_W=statistics.mean(x[3] for x in xs))
  result['gpu_windows']=windows
 results[label]=result;cost+=expected
assert grid=={(lam,st) for lam in [.05,.2] for st in range(6,10)} and cost==18584
out=dict(status='PASS_LOCAL_PARALLEL_DIAGNOSTICS',observed=js(inputs[-1][1]/'EVIDENCE.json')['observed'],cases=results,covered_combinations=sorted(grid),parallel_diagnostic_optimizer_calls=cost,parallel_diagnostic_extra_autograd=0,prior_engineering_optimizer_calls=15965,total_engineering_optimizer_calls=34549,total_engineering_extra_autograd=83,environments=0,production_epochs_added=0,abandoned_partial_recorded_optimizer_calls=140,abandoned_partial_unrecorded_cost='UNKNOWN',limits=['Diagnostic paths use immutable accepted-S checkpoints/cache as test fixtures, not replacement production trajectories or caches.','Timing is full Task9 epoch compute for two workers under continuing production background; excludes checkpoint/collection/preparation; not whole-run acceleration.','Only exact equality at exercised full epochs and state paths; all future stages still require their own admissions.','No MPS or GPU configuration changes; no production training signals.'])
(E/'PARALLEL_TRIAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=out['status'],parallel_optimizer=cost,total_engineering=34549,timing=results['timing']['reports'][0])))
