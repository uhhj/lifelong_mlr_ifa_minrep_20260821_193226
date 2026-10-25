"""Versioned dispatcher handoff; unaffected D continues without restarting."""
import os,subprocess,json,time,hashlib,signal,ast,shutil
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_FULL_GOAL41_20261025_v1_engineering02')
OLD=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_FULL_GOAL41_20261025_v1')
PD=Path('/data/mlr_ifa/experiments/CCBP_V4_FULL_GOAL41_20261025_v1')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 pp=Path('/proc')/str(pid);raw=(pp/'stat').read_text();f=raw[raw.rfind(')')+2:].split();cmd=(pp/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 return dict(state=f[0],starttime=f[19],cmd=cmd)
assert not Q.exists(),'No engineering02 redeployment'
assert sha(OLD/'CONTRACT.json')=='8edb70c911328d8440f25dea66f719c8259a4c332bef5be5a55caec84aea452c'
oc=json.loads((OLD/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in oc['files'].items()) and all(sha(S/n)==h for n,h in oc['parent_source'].items())
assert not (OLD/'FAILURE.json').exists() and not (OLD/'controller.exit').exists() and not (PD/'O').exists() and not (PD/'S').exists()
ctrl=proc(18453);worker=proc(18454);assert str(OLD/'controller.py') in ctrl['cmd'] and 'torch.distributed.run' in worker['cmd'] and worker['cmd'].strip().endswith('train.py D')
assert ctrl['state'] not in ['T','t','Z'] and worker['state']!='Z'
hist=json.loads((OLD/'EPOCHS_D.json').read_text());assert hist and hist[-1]['stage']<=9
assert all(len(json.loads(p.read_text())['rows'])<=128 for p in OLD.glob('MEMORY_D_*.json'))
for mount,reserve,root in [('/root',1024**3,False),('/data',2*1024**3,True)]:
 st=os.statvfs(mount);free=(st.f_bfree if root else st.f_bavail)*st.f_frsize;assert free-(4*1024**3 if root else 0)>=reserve
import torch
source=S/'caet_v3_exp/model.py';fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='feature');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
observed=[]
for length in range(1,9):
 try:ns['feature'](torch.zeros(1,length,4,512));accepted=True
 except AssertionError:accepted=False
 assert accepted==(length==8);observed.append(dict(length=length,legacy_accepted=accepted))
for n,s in BUNDLE.items():
 assert Path(n).name==n and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir();files=dict(oc['files'])
for n,s in BUNDLE.items():(Q/n).write_text(s);files[str(Q/n)]=sha(Q/n)
files[str(OLD/'CONTRACT.json')]=sha(OLD/'CONTRACT.json')
shutil.copyfile(OLD/'INITIAL_D.json',Q/'INITIAL_D.json');files[str(Q/'INITIAL_D.json')]=sha(Q/'INITIAL_D.json')
transition=dict(old_root=str(OLD),old_controller=18453,old_torchrun=18454,controller_starttime=ctrl['starttime'],worker_starttime=worker['starttime'],observed_D_epochs=len(hist),observed_D_steps=hist[-1]['steps'],only_dispatch_paused=True,no_D_restart=True)
(Q/'TRANSITION_INPUT.json').write_text(json.dumps(transition,indent=2));files[str(Q/'TRANSITION_INPUT.json')]=sha(Q/'TRANSITION_INPUT.json')
(Q/'LEGACY_LENGTH_REPRODUCTION.json').write_text(json.dumps(dict(status='REPRODUCED_INTERFACE_DEFECT',source_sha256=sha(source),lengths=observed,model_calls=0,optimizer_calls=0,environments=0),indent=2));files[str(Q/'LEGACY_LENGTH_REPRODUCTION.json')]=sha(Q/'LEGACY_LENGTH_REPRODUCTION.json')
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,source_root=str(S),parent_source=oc['parent_source'],official_initialization=oc['official_initialization'],inherited_D_contract=sha(OLD/'CONTRACT.json'),inherited_D_root=str(OLD),inheritance='Only original D uses original contract; observed natural exit required; no completed epoch or collection replay',created=time.time(),budget=oc['budget'],capacity=oc['capacity']),indent=2))
r=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-c','import runtime,memory,updates,initialization,train,collect,evaluate; import synthetic_checks; synthetic_checks.run()'],cwd=Q,capture_output=True,text=True)
(Q/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)));assert r.returncode==0,(r.stdout,r.stderr)
assert proc(18453)['starttime']==ctrl['starttime'] and proc(18454)['starttime']==worker['starttime'];os.kill(18453,signal.SIGSTOP)
for _ in range(20):
 if proc(18453)['state'] in ['T','t']:break
 time.sleep(.05)
assert proc(18453)['state'] in ['T','t'];assert proc(18454)['state'] not in ['T','t','Z']
(Q/'DISPATCH_PAUSE.json').write_text(json.dumps(dict(time=time.time(),controller=proc(18453),uninterrupted_D_worker=proc(18454)),indent=2))
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='VERSIONED_DISPATCH_HANDOFF_D_CONTINUES',root=str(Q),controller=p.pid,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),original_D_uninterrupted=transition,short_history_gpu_admission='PENDING_OWN_OS',observed=time.time())))
