"""Run serial then concurrent isolated workers, never manipulate production."""
import os,sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
stages=[6,7,8,9];lambdas=[.05,.2];children=[];events=[];metrics=[]
def js(p):return json.loads(p.read_text())
def put(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def production():return {c:js(Q/c/'TRAIN_STATUS.json') for c in ['L005_41','L005_42','L005_43'] if (Q/c/'TRAIN_STATUS.json').exists()}
def tick():
 p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
 metrics.append(dict(time=time.time(),gpu=p.stdout.strip(),exit=p.returncode))
 put(D/'METRICS.json',metrics)
def guard_capacity():
 mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert p.returncode==0
 assert mem['MemAvailable']>=12*1024**2 and float(p.stdout.strip())>=10000,'Insufficient diagnostic headroom'
def launch(seed,mode):
 target=D/(mode+'_'+str(seed));assert not target.exists();target.mkdir()
 with (target/'worker.log').open('x') as f:
  p=subprocess.Popen([sys.executable,'-B',str(D/'worker.py'),str(target),str(seed),mode,','.join(map(str,stages)),','.join(map(str,lambdas))],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 children.append((p,target));events.append(dict(event='launch',seed=seed,mode=mode,pid=p.pid,time=time.time(),production=production()));put(D/'EVENTS.json',events)
 return p,target
def phase(workers,mode):
 for lam in lambdas:
  for stage in stages:
   if lam==.05 and stage==9:continue
   label=str(lam)+'_'+str(stage);start=time.time()
   while not all((t/(label+'_READY.json')).exists() for p,t in workers):
    assert all(p.poll() is None for p,t in workers),'Worker exited before READY'
    assert time.time()-start<900;tick();time.sleep(2)
   put(D/(mode+'_'+label+'_GO.json'),dict(time=time.time()))
   while not all((t/(label+'_DONE.json')).exists() for p,t in workers):
    assert all(p.poll() in [None,0] for p,t in workers),'Worker failed'
    assert time.time()-start<1800;tick();time.sleep(2)
 for p,t in workers:assert p.wait(timeout=120)==0 and (t/'RESULT.json').exists()
 events.append(dict(event='phase_complete',mode=mode,time=time.time(),production=production()));put(D/'EVENTS.json',events)
def main():
 assert not (D/'CONTINUATION_STARTED.json').exists(),'single use';put(D/'CONTINUATION_STARTED.json',dict(pid=os.getpid(),time=time.time(),scope='Seven remaining lambda/stage combinations, independent seeds42/43; production continues'))
 global events,metrics
 events=js(D/'EVENTS.json');metrics=js(D/'METRICS.json')
 assert not any(e['event']=='launch' and e['mode']=='parallel' for e in events)
 assert all(js(D/('serial'+str(seed)+'_'+str(seed))/'RESULT.json')['optimizer_completed']==3196 for seed in [42,43])
 guard_capacity();phase([launch(42,'parallel'),launch(43,'parallel')],'parallel')
 reports=[]
 for lam in lambdas:
  for stage in stages:
   if lam==.05 and stage==9:continue
   label=str(lam)+'_'+str(stage);serial=[];parallel=[]
   for seed in [42,43]:
    ref=D/('serial'+str(seed)+'_'+str(seed));new=D/('parallel_'+str(seed));a=js(ref/(label+'_TRACE.json'));b=js(new/(label+'_TRACE.json'));assert a==b,('Parallel trajectory mismatch',seed,label)
    serial.append(js(ref/(label+'_DONE.json')));parallel.append(js(new/(label+'_DONE.json')))
   s=sum(x['elapsed_seconds'] for x in serial);p=max(x['ended'] for x in parallel)-min(x['started'] for x in parallel)
   reports.append(dict(lambda_R=lam,stage=stage,all_per_step_scalars_branch_state_digests_probes_decisions_and_next_step_exact=True,serial_two_worker_seconds=s,parallel_two_worker_seconds=p,aggregate_throughput_ratio=s/p,wall_reduction_fraction=1-p/s,serial_compute_spans=[x['compute_span_seconds'] for x in serial],parallel_compute_spans=[x['compute_span_seconds'] for x in parallel],limitation='Includes diagnostic full-state digest overhead; current production remained as background. Not a pure one-versus-two production A/B.'))
 results=[js(t/'RESULT.json') for p,t in children];current=sum(x['optimizer_completed'] for x in results);assert current==6392 and all(x['optimizer_started']==x['optimizer_completed'] for x in results)
 retained=[js(D/('serial'+str(seed)+'_'+str(seed))/'RESULT.json') for seed in [42,43]];assert all(x['optimizer_started']==x['optimizer_completed']==3196 for x in retained);calls=current+sum(x['optimizer_completed'] for x in retained)
 put(D/'RESULT.json',dict(status='PASS_REMAINING_SEVEN_STAGE_PARALLEL_EXACT',optimizer_started=calls,optimizer_completed=calls,extra_autograd=0,environments=0,production_epochs=0,production_writes=False,production_signals=0,ended=time.time(),reports=reports,continuation_current_optimizer_calls=current,retained_serial_optimizer_calls=6392,resolved_guard_failure_sha256=hashlib.sha256((D/'FAILURE.json').read_bytes()).hexdigest(),continuation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),admission='Together with initial stage9 .05 probe covers both lambdas all stages; still requires scheduler/environment isolation review and actual production admissions'))
if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc()
  for p,t in children:
   if p.poll() is None:p.terminate()
  put(D/'CONTINUATION_FAILURE.json',dict(time=time.time(),error=error,production_signals=0,diagnostic_children=[dict(pid=p.pid,path=str(t),returncode=p.poll()) for p,t in children],unpersisted_partial_diagnostic_cost='UNKNOWN_IF_WORKER_FAILED_IN_BLOCK'))
  raise
