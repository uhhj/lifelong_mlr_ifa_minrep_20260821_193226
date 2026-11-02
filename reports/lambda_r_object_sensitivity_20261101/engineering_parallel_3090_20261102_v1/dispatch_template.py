"""Single-use natural-boundary dispatcher. SOURCES is inserted by the local builder."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
E=Q/'engineering_parallel_3090_20261102_v1'
SOURCES={}
EXPECTED_SOURCES={}
EXPECTED_CONTINUATION='__EXPECTED_CONTINUATION__'
assert set(SOURCES)=={'controller.py','train_wrapper.py','watchdog.py'},'Template is not executable; use reviewed embedded dispatcher'
assert set(EXPECTED_SOURCES)=={'probe','timing','full'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
assert not E.exists(),'Never repeat engineering dispatch'
proofs={};frozen={};baseline={};grid=set();cost=0
specs=[('probe','lambda_parallel_stage9_20261102_v1','PASS_STAGE9_PARALLEL_EXACT_FEASIBILITY',2904),('timing','lambda_parallel_timing_20261102_v1','PASS_STAGE9_PARALLEL_THROUGHPUT_EXACT_ENDSTATE',2896),('full','lambda_parallel_full_20261102_v1','PASS_REMAINING_SEVEN_STAGE_PARALLEL_EXACT',12784)]
for label,name,status,calls in specs:
 d=Path('/data/mlr_ifa/diagnostics')/name;v=js(d/'RESULT.json')
 assert v['status']==status and v['optimizer_started']==v['optimizer_completed']==calls
 assert v['extra_autograd']==v['environments']==v['production_epochs']==0 and not v['production_writes'] and v['production_signals']==0
 if label=='full':
  continuation=js(d/'CONTINUATION_DISPATCH.json');failure=js(d/'FAILURE.json')
  assert 'Insufficient diagnostic headroom' in failure['error'] and len(failure['diagnostic_children'])==2 and all(x['returncode']==0 for x in failure['diagnostic_children'])
  assert sha(d/'FAILURE.json')==continuation['prior_guard_failure_sha256']==v['resolved_guard_failure_sha256']
  assert sha(d/'continue_parallel_v2.py')==continuation['source_sha256']==v['continuation_source_sha256']==EXPECTED_CONTINUATION
  identity=proc(continuation['pid']);assert identity is None or identity['state']=='Z'
  assert v['continuation_current_optimizer_calls']==v['retained_serial_optimizer_calls']==6392
  assert all(sha(p)==h for p,h in continuation['retained_serial_result_hashes'].items()) and not (d/'CONTINUATION_FAILURE.json').exists()
  for source,target in [('continue_parallel_v2.py','full_continue_parallel_v2.py'),('FAILURE.json','full_prior_guard_FAILURE_EVIDENCE.json'),('CONTINUATION_DISPATCH.json','full_CONTINUATION_DISPATCH.json')]:frozen[target]=(d/source).read_text()
 else:assert not (d/'FAILURE.json').exists()
 dispatch=js(d/'DISPATCH.json');identity=proc(dispatch['pid']);assert identity is None or identity['state']=='Z'
 manifest=js(d/'SOURCE_MANIFEST.json');assert manifest==dispatch['files']==EXPECTED_SOURCES[label] and all(sha(p)==h for p,h in manifest.items())
 for event in js(d/'EVENTS.json'):
  if event['event']=='launch':
   identity=proc(event['pid']);assert identity is None or identity['state']=='Z'
 worker_results=sorted(d.glob('*/RESULT.json'));assert len(worker_results)==4
 for worker in worker_results:
  w=js(worker);assert w['optimizer_started']==w['optimizer_completed'] and w['environments']==0
  assert w['optimizer_completed']==calls//4
  hashes=js(worker.parent/'PROGRESS.json')['baseline_hashes'];assert all(sha(p)==h for p,h in hashes.items());baseline.update(hashes)
  frozen[label+'_'+worker.parent.name+'_RESULT.json']=worker.read_text()
 if label=='full':
  assert len(v['reports'])==7
  for report in v['reports']:
   assert report['all_per_step_scalars_branch_state_digests_probes_decisions_and_next_step_exact'];grid.add((report['lambda_R'],report['stage']))
 else:
  assert len(v['reports'])==1;grid.add((.05,9))
 if label=='probe':assert v['reports'][0]['all_per_step_scalars_branch_state_digests_probes_decisions_and_next_step_exact']
 if label=='timing':assert v['reports'][0]['aggregate_throughput_ratio']>=1.1 and v['reports'][0]['per_step_scalars_probes_decisions_initial_final_state_exact']
 proofs[label]=dict(directory=str(d),result_sha256=sha(d/'RESULT.json'),source_manifest=manifest,optimizer_calls=calls,worker_result_hashes={str(p):sha(p) for p in worker_results})
 frozen[label+'_RESULT.json']=(d/'RESULT.json').read_text()
 for file in ['worker.py','driver.py']:frozen[label+'_'+file]=(d/file).read_text()
 cost+=calls
assert grid=={(lam,st) for lam in [.05,.2] for st in range(6,10)} and cost==18584
for name in ['CONTRACT.json','engineering_device_snapshot_20261101_v1/CONTRACT.json','engineering_resume_3090_20261102_v1/CONTRACT.json','engineering_checks_3090_20261102_v1/CONTRACT.json']:
 assert all(sha(p)==h for p,h in js(Q/name)['files'].items())
old=proc(4347);child=proc(4689);training=proc(4694)
assert old and old['state']=='S' and old['cmd'].strip().endswith(str(Q/'engineering_checks_3090_20261102_v1/controller.py'))
assert child and 'torch.distributed.run' in child['cmd'] and training and str(Q/'engineering_checks_3090_20261102_v1/train_wrapper.py') in training['cmd']
status=js(Q/'STATUS.json');assert status['phase']=='train' and status['suite']=='L005' and status['seed']==42 and status['pid']==4689
for name in ['controller.exit','engineering_device_snapshot_20261101_v1/controller.exit','engineering_resume_3090_20261102_v1/controller.exit','engineering_checks_3090_20261102_v1/controller.exit']:assert not (Q/name).exists()
for case in ['L005_43','L020_41','L020_42','L020_43']:assert not (Q/case/'train_S.log').exists()
finished=Q/'L005_41';assert (finished/'train_S.exit').read_text().strip()=='0';t=js(finished/'TRAIN_S.json')
assert t['epochs']==400 and t['training_collection_episodes']==40 and t['physical_optimizer_calls']==195300
completed={str(p):sha(p) for p in finished.glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']}
# The complete first-run checkpoint set is outside the lightweight workspace.
private=Path('/data/mlr_ifa/experiments')/Q.name/'L005_41'/'S'
assert all((private/('stage%d.pth'%st)).exists() for st in range(6,10))
assert all(sha(private/('stage%d.pth'%st))==t['checkpoints'][str(st)] for st in range(6,10))
completed.update({str(private/('stage%d.pth'%st)):sha(private/('stage%d.pth'%st)) for st in range(6,10)})
raw=(Q/'L005_42/GATES.jsonl').read_bytes();raw=raw[:raw.rfind(b'\n')+1];history=js(Q/'L005_42/EPOCHS_S.json')
guard=dict(time=time.time(),controller=old,child=child,training=training,queue_status=status,epoch_count=len(history),epoch_prefix_sha256=hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest(),gate_prefix_bytes=len(raw),gate_prefix_sha256=hashlib.sha256(raw).hexdigest(),completed_L005_41_hashes=completed,training_signals=0)
admission=dict(status='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING',time=time.time(),proofs=proofs,covered_stages=sorted(grid),optimizer_started=cost,optimizer_completed=cost,extra_autograd=0,environments=0,production_epochs=0,baseline_hashes=baseline,scope='Serial versus concurrent two seeds, full epoch per lambda/stage; not proof of all future 400-epoch trajectories',max_training_concurrency=2,collections='serialized lock',formal_evaluation='serial')
E.mkdir()
for n,s in {**SOURCES,**frozen}.items():(E/n).write_text(s)
(E/'PARALLEL_ADMISSION.json').write_text(json.dumps(admission,indent=2));(E/'GUARD.json').write_text(json.dumps(guard,indent=2))
names=[*SOURCES,*frozen,'PARALLEL_ADMISSION.json','GUARD.json']
contract=dict(version=E.name,scientific_contract_sha256=sha(Q/'CONTRACT.json'),files={str(E/n):sha(E/n) for n in names},change='Only independent future training scheduling, maximum two; unchanged previous exact training wrapper; serialized training collection and formal evaluation',diagnostic_optimizer_calls=34549,diagnostic_extra_autograd=83,prior_diagnostic_optimizer_calls=15965,parallel_diagnostic_optimizer_calls=18584,parallel_diagnostic_extra_autograd=0,diagnostic_environments=0,abandoned_partial_recorded_optimizer_calls=140,abandoned_partial_unrecorded_cost='UNKNOWN')
(E/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
with (E/'controller.log').open('x') as f:p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
receipt=dict(status='ENGINEERING_DISPATCHED_NOT_YET_HANDOFF',time=time.time(),pid=p.pid,contract_sha256=sha(E/'CONTRACT.json'),contract=contract)
(E/'DISPATCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
