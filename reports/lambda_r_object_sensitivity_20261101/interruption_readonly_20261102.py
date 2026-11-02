"""CPU-only inspection after retained pause processes disappeared; never resumes."""
import os,json,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');E=Q/'engineering_device_snapshot_20261101_v1'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
pause=js(Q/'user_pause_20261101_v1/STATUS.json')
same={f:sha(Q/f)==h for f,h in pause['light_hashes'].items()}
assert all(same.values()),'Paused metadata changed'
contracts={str(c):dict(sha256=sha(c),files=len(js(c)['files']),exact=all(sha(Path(f))==h for f,h in js(c)['files'].items())) for c in [Q/'CONTRACT.json',E/'CONTRACT.json']}
assert all(x['exact'] for x in contracts.values())
private=Path('/data/mlr_ifa/experiments')/Q.name
cps=list(private.rglob('recovery.pth'));assert len(cps)==1,[str(p) for p in cps]
import torch
torch.set_num_threads(1)
cp=cps[0];x=torch.load(cp,map_location='cpu')
h=js(Q/'L005_41/EPOCHS_S.json');g=[json.loads(s) for s in (Q/'L005_41/GATES.jsonl').read_text().splitlines()]
assert x['history']==h and len(h)==260 and x['stage']==8 and x['next_epoch']==60
assert x['steps']==56940 and x['physical_optimizer_calls']==98180 and x['gate_blocks']==h[-1]['gate_blocks']
assert {'model','optimizer','scheduler','grads','modes','requires','rng','step'}<=x['state'].keys()
assert all(torch.equal(x['state']['model'][k],v) for k,v in x['state_dict'].items())
assert x['probe_sha256']==sha(Q/'L005_41/PROBES_8.json')
tail=g[x['gate_blocks']:]
assert all(z['stage']==8 and z['epoch']==60 for z in tail)
uptime=float(Path('/proc/uptime').read_text().split()[0]);now=time.time()
out=dict(status='INTERRUPTED_NO_IN_MEMORY_RESUME_CPU_CHECKPOINT_VERIFIED',observed=now,uptime_seconds=uptime,boot_time_estimate=now-uptime,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),paused_time=pause['time'],boot_after_pause=now-uptime>pause['time'],contracts=contracts,paused_light_files_exact=len(same),prior_pids={k:Path('/proc',k).exists() for k in pause['after']},gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),gpu_processes=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),checkpoint=dict(path=str(cp),sha256=sha(cp),bytes=cp.stat().st_size,stage=x['stage'],next_epoch_zero_based=x['next_epoch'],completed_epochs=len(h),steps=x['steps'],physical_optimizer_calls=x['physical_optimizer_calls'],gate_blocks=x['gate_blocks'],full_state_fields=list(x['state']),rng_fields=list(x['state']['rng']) if isinstance(x['state']['rng'],dict) else type(x['state']['rng']).__name__,history_exact=True,model_exact=True,probe_exact=True,contract=x['contract']),lost_partial=dict(recorded_gate_blocks=len(tail),accepted_batch_updates=sum(z['steps'] for z in tail),recorded_candidate_optimizer_calls=2*sum(z['steps'] for z in tail),gate_records=tail,unrecorded_last_block_cost='UNKNOWN',state_available=False),exit_files={str(p):p.read_text() if p.exists() else None for p in [Q/'controller.exit',E/'controller.exit',Q/'L005_41/train_S.exit']},cpu_diagnostic_optimizer_calls=0,environments=0,signals=0,production_writes=0)
print(json.dumps(out))
