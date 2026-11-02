"""Unique isolated launcher/recovery validation dispatch; never production restart."""
import os,sys,json,time,hashlib,subprocess,shutil,uuid
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');D=Path('/data/mlr_ifa/diagnostics/rendezvous_recovery_20261102_v1');R=Q/'L005_43';P=Path('/data/mlr_ifa/experiments')/Q.name/'L005_43/S';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();js=lambda p:json.loads(Path(p).read_text())

assert D.exists() and not (D/'DISPATCH.json').exists() and not (D/'LEDGER.json').exists()
assert not (D/'A_v2.log').exists() and not (D/'B_v2.log').exists()
assert all('unrecognized arguments' in (D/(x+'.log')).read_text() for x in ['A','B'])
binding=js(D/'BINDING.json');assert all(sha(D/n)==h for n,h in binding['source_hashes'].items()) and sha(P/'recovery.pth')==binding['checkpoint_sha256']
(D/'STARTUP_ATTEMPT1_FAILURE.json').write_text(json.dumps(dict(cause='Installed torch1.13 accepts underscore option spelling',log_hashes={x:sha(D/(x+'.log')) for x in ['A','B']},optimizer=0,environments=0,validator_never_started=True),indent=2))
def command(script,*args):return [sys.executable,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),'--monitor_interval=1',str(script),*map(str,args)]
children=[]
for label,seconds in [('A',5),('B',18)]:
 with (D/(label+'_v2.log')).open('x') as f:p=subprocess.Popen(command(D/'probe_worker.py',D/(label+'_v2.json'),seconds),stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 children.append((label,p))
rc={label:p.wait(timeout=60) for label,p in children};assert all(v==0 for v in rc.values()),rc
a=js(D/'A_v2.json');b=js(D/'B_v2.json');assert a['master_port']!=b['master_port'] and b['end']>a['end']+8 and all(x['world_size']=='1' and x['rank']=='0' for x in [a,b])
proof=dict(status='PASS_INDEPENDENT_EPHEMERAL_RENDEZVOUS_LIFECYCLE',results=dict(A=a,B=b),returncodes=rc,early_peer_exit_survived=True,optimizer_calls=0,environments=0,production_signals=0)
(D/'LAUNCHER_VALIDATION.json').write_text(json.dumps(proof,indent=2))
with (D/'validation.log').open('x') as f:p=subprocess.Popen(command(D/'validate.py'),cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
r=dict(status='ISOLATED_RECOVERY_VALIDATION_STARTED',pid=p.pid,time=time.time(),source_hashes={n:sha(D/n) for n in ['validate.py','probe_worker.py']},planned_optimizer=869,planned_extra_autograd=11,environments=0,launcher_validation=proof)
(D/'DISPATCH.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
