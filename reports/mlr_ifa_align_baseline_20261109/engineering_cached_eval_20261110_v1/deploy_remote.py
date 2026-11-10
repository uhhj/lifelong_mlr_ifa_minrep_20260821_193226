"""Unique reviewed boundary owner; never signal an evaluator or trainer."""
from pathlib import Path
import sys,json,hashlib,subprocess,time,os
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_cached_eval_20261110_v1';OLD=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(OLD));import resume_runtime as rt
rt.verify();assert not E.exists() and not (OLD/'controller.exit').exists()
v=json.loads(Path('/data/mlr_ifa/diagnostics/align_accepted_cache_validation_20261110_v1/VALIDATION.json').read_text())
assert v['status']=='PASS_OFFLINE_ACCEPTED_CACHE_ADAPTATION'
assert v['files']['cache_policy.py']==hashlib.sha256(BUNDLE['cache_policy.py'].encode()).hexdigest()
for n,t in BUNDLE.items():
    assert not Path(n).is_absolute() and '..' not in Path(n).parts
    if n.endswith('.py'):compile(t,n,'exec')
oldpid=40721;cmd=(Path('/proc')/str(oldpid)/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(OLD/'controller_resume.py') in cmd
def proc(pid):
    s=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]))
ev=[p for p in rt.processes([Q,rt.P]) if str(OLD/'evaluate_guarded.py') in p['cmd']];assert len(ev)==1,ev
case=ev[0]['cmd'].strip().split()[-1];assert case in ['OBJECT_41','OBJECT_42','OBJECT_43']
child=proc(ev[0]['pid']);assert child['state'] not in ['T','Z','X']
assert int((Path('/proc')/str(child['pid'])/'stat').read_text().split(') ',1)[1].split()[1])==oldpid
assert not any((rt.P/('GOAL_'+str(s))/'train.exit').exists() for s in [41,42,43])
prefix={};immutable={}
for c in ['OBJECT_41','OBJECT_42','OBJECT_43']:
    assert rt.js(rt.P/c/'random/TRAINING_COMPLETE.json')['epochs']==400
    for p in (rt.P/c).rglob('*'):
        if not p.is_file():continue
        if '/evaluation/' in str(p):
            if c==case and p.suffix=='.jsonl':
                raw=p.read_bytes();raw=raw[:raw.rfind(b'\n')+1];prefix[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            elif c!=case:immutable[str(p)]=rt.sha(p)
        elif p.suffix in ['.pth','.pt'] or p.name in ['epochs.json','OPTIMIZER_CALLS.json','INITIAL_STATE.json','TRAINING_COMPLETE.json'] or p.name.startswith('ADMISSION_'):immutable[str(p)]=rt.sha(p)
g=dict(observed=time.time(),case=case,old_controller=proc(oldpid),evaluator=child,prefix_files=prefix,immutable_files=immutable)
capacity=rt.original.capacity();E.mkdir()
for n,t in BUNDLE.items():
    with (E/n).open('x') as f:f.write(t)
rt.write(E/'GUARD.json',g);rt.write(E/'ENGINEERING_VALIDATION.json',v)
files={n:rt.sha(E/n) for n in BUNDLE};files.update({n:rt.sha(E/n) for n in ['GUARD.json','ENGINEERING_VALIDATION.json']})
rt.write(E/'CONTRACT.json',dict(status='FROZEN_ACCEPTED_CACHE_BOUNDARY_ENGINEERING',files=files,science_contract=rt.sha(Q/'CONTRACT.json'),old_engineering_contract=rt.sha(OLD/'CONTRACT.json'),cache_tolerance=dict(atol=1e-4,rtol=1e-4),global_rng_and_normal_sampling_unchanged=True,completed_episodes_retained=True))
with (E/'controller.log').open('x') as f:p=subprocess.Popen([rt.PY,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rt.write(E/'DEPLOYMENT.json',dict(observed=time.time(),controller=p.pid,contract_sha256=rt.sha(E/'CONTRACT.json'),capacity=capacity,status='STARTED_PENDING_ARMED_AND_BOUNDARY'))
print(json.dumps(dict(status='STARTED_PENDING_ARMED_AND_BOUNDARY',controller=p.pid,case=case,active_evaluator=child,contract_sha256=rt.sha(E/'CONTRACT.json'),files=files,new_optimizer=0,new_environment=0)))
