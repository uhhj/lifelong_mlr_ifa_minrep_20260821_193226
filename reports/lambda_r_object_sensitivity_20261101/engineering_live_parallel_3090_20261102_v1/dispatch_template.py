"""Unique live adoption, no signal or restart to the current training process."""
import os,sys,json,time,hashlib,subprocess,site
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');P=Q/'engineering_parallel_3090_20261102_v1';E=Q/'engineering_live_parallel_3090_20261102_v1'
SOURCES={}
assert set(SOURCES)=={'controller.py','watchdog.py','train_wrapper.py','collection_startup_hook.py'},'Not an executable template'
js=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
assert not E.exists(),'Never repeat dispatch'
for name in ['CONTRACT.json','engineering_device_snapshot_20261101_v1/CONTRACT.json','engineering_resume_3090_20261102_v1/CONTRACT.json','engineering_checks_3090_20261102_v1/CONTRACT.json','engineering_parallel_3090_20261102_v1/CONTRACT.json']:
 assert all(sha(p)==h for p,h in js(Q/name)['files'].items())
admission=js(P/'PARALLEL_ADMISSION.json');assert admission['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING' and admission['optimizer_completed']==18584
proof=Path('/data/mlr_ifa/diagnostics/live_parallel_lock_20261102_v3/RESULT.json');v=js(proof)
assert v['status']=='PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK' and v['random_values_and_sys_path_exact'] and v['unmatched_interpreter_unaffected'] and v['disabled_hook_unaffected'] and v['temporary_pth_removed']
assert v['optimizer_calls']==v['environments']==v['production_signals']==0 and v['hook_sha256']==hashlib.sha256(SOURCES['collection_startup_hook.py'].encode()).hexdigest()
assert not (P/'HANDOFF_COMPLETE.json').exists() and not (P/'FAILURE.json').exists() and not (P/'WATCHDOG_RELEASE.json').exists()
controllers={k:proc(pid) for k,pid in [('root',4347),('parallel',8608),('watchdog',8609)]}
assert controllers['root']['state']=='T' and all(controllers[k]['state'] in ['S','R','D'] for k in ['parallel','watchdog'])
assert str(P/'controller.py') in controllers['parallel']['cmd'] and str(P/'watchdog.py') in controllers['watchdog']['cmd']
child=proc(4689);training=proc(4694);assert child['state']!='Z' and training['state'] not in ['T','Z','X']
assert str(Q/'engineering_checks_3090_20261102_v1/train_wrapper.py') in training['cmd'] and 'L005_42' in training['cmd']
current=js(Q/'L005_42/TRAIN_STATUS.json');assert current['phase']=='train' and current['pid']==4694 and current['stage'] in [8,9] and current['epoch_complete']<90
for case in ['L005_43','L020_41','L020_42','L020_43']:assert not (Q/case/'train_S.log').exists()
for name in ['controller.exit','engineering_device_snapshot_20261101_v1/controller.exit','engineering_resume_3090_20261102_v1/controller.exit','engineering_checks_3090_20261102_v1/controller.exit','engineering_parallel_3090_20261102_v1/controller.exit']:assert not (Q/name).exists()
mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1]);gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert gpu.returncode==0 and mem>=12*1024**2 and float(gpu.stdout.strip())>=10000
pth=Path(site.getsitepackages()[0])/'zzz_mlr_lambda_live_collection_20261102.pth';assert not pth.exists()
targets={str(Q/case/'collect.py'):case for case in ['L005_42','L005_43','L020_41','L020_42','L020_43']}
body="try:\n _hook_path="+repr(str(E/'collection_startup_hook.py'))+"\n exec(compile(open(_hook_path).read(),_hook_path,'exec'),{'__file__':_hook_path,'__name__':'_mlr_startup_hook'})\nexcept BaseException:\n import os,traceback\n try:traceback.print_exc()\n finally:os._exit(86)\n"
line='import sys,os; exec('+repr(body)+') if sys.argv[0] in '+repr(tuple(targets))+' and os.path.isfile('+repr(str(E/'HOOK_ENABLED.json'))+') else None\n'
g=js(P/'GUARD.json');assert all(sha(p)==h for p,h in g['completed_L005_41_hashes'].items())
h=js(Q/'L005_42/EPOCHS_S.json');raw=(Q/'L005_42/GATES.jsonl').read_bytes();raw=raw[:raw.rfind(b'\n')+1]
expected={case:[st for st in range(6,10) if not (Q/case/('MEMORY_S_%d.json'%st)).exists()] for case in targets.values()}
guard=dict(time=time.time(),controllers=controllers,child=child,training=training,epoch_count=len(h),epoch_prefix_sha256=hashlib.sha256(json.dumps(h,sort_keys=True,separators=(',',':')).encode()).hexdigest(),gate_prefix_bytes=len(raw),gate_prefix_sha256=hashlib.sha256(raw).hexdigest(),completed_L005_41_hashes=g['completed_L005_41_hashes'],expected_collection_stages=expected,prior_parallel_contract=sha(P/'CONTRACT.json'),training_signals=0)
E.mkdir()
for n,s in SOURCES.items():(E/n).write_text(s)
(E/'HOOK_CONFIG.json').write_text(json.dumps(dict(targets=targets,lock_path='/data/mlr_ifa/'+Q.name+'_collection.lock'),indent=2))
(E/'GUARD.json').write_text(json.dumps(guard,indent=2));(E/'STARTUP_LOCK_VALIDATION.json').write_bytes(proof.read_bytes());(E/'startup_hook.pth.txt').write_text(line)
temporary=pth.with_suffix('.pending');assert not temporary.exists()
with temporary.open('x') as f:f.write(line);f.flush();os.fsync(f.fileno())
os.link(temporary,pth);temporary.unlink() # atomic, exclusive; inert until HOOK_ENABLED
files={str(E/n):sha(E/n) for n in [*SOURCES,'HOOK_CONFIG.json','GUARD.json','STARTUP_LOCK_VALIDATION.json','startup_hook.pth.txt']};files[str(pth)]=sha(pth)
contract=dict(version=E.name,files=files,hook_files={str(p):sha(p) for p in [E/'collection_startup_hook.py',E/'HOOK_CONFIG.json',pth]},scientific_contract_sha256=sha(Q/'CONTRACT.json'),prior_parallel_contract_sha256=sha(P/'CONTRACT.json'),change='Start L00543 alongside unchanged live L00542; exact-path pre-import collection locks; no production math/data changes',diagnostic_optimizer_calls=34549,diagnostic_extra_autograd=83,additional_startup_validation_optimizer=0,additional_startup_validation_environments=0,global_startup_file=str(pth),startup_targets=targets,expected_collection_stages=expected)
(E/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
with (E/'controller.log').open('x') as f:p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
r=dict(status='LIVE_PARALLEL_DISPATCHED_PENDING_PROCESS_AUDIT',time=time.time(),pid=p.pid,contract_sha256=sha(E/'CONTRACT.json'),contract=contract,training_signals=0)
(E/'DISPATCH.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
