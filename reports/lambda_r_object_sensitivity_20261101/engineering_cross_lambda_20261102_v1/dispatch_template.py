"""Single guarded live handoff, authorizing only one cross-lambda early seed."""
import os,sys,json,time,hashlib,subprocess,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');E=Q/'engineering_cross_lambda_20261102_v1';PREV=Q/'engineering_rendezvous_recovery_20261102_v1';D=Path('/data/mlr_ifa/diagnostics/cross_lambda_environment_lock_20261102_v1')
SOURCES={}
js=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,state=s[0],start=s[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
assert not E.exists() and not D.exists(),'Never repeat deployment or zero-environment check'
assert set(SOURCES)=={'controller.py','watchdog.py','evaluate_locked.py'}
for p in [Q/'CONTRACT.json',*Q.glob('engineering_*/CONTRACT.json')]:assert all(sha(n)==h for n,h in js(p)['files'].items())
assert sha(PREV/'CONTRACT.json')=='23f1dda97de2ebd9a1b752db2f34ab23ccd4e2f4ed9461ce4e0d99887a5fa7c4'
identities={k:proc(pid) for k,pid in [('old_controller',11239),('child',11241),('training',11246)]}
assert all(x['state'] in ['S','R','D'] for x in identities.values())
assert '/controller.py' in identities['old_controller']['cmd'] and str(PREV) in identities['old_controller']['cmd']
assert 'torch.distributed.run' in identities['child']['cmd'] and '--rdzv_endpoint=localhost:0' in identities['child']['cmd']
assert 'L005_43' in identities['training']['cmd'] and 'torch.distributed.run' not in identities['training']['cmd']
assert not (PREV/'controller.exit').exists() and not (PREV/'L005_43_resume.exit').exists()
for case in ['L020_41','L020_42','L020_43']:assert not (Q/case/'train_S.log').exists()
for case in ['L005_41','L005_42','L005_43']:assert not (Q/case/'evaluate.log').exists()
assert js(Q/'engineering_parallel_3090_20261102_v1/PARALLEL_ADMISSION.json')['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
assert js(PREV/'LAUNCHER_VALIDATION.json')['early_peer_exit_survived']
assert (Q/'engineering_live_parallel_3090_20261102_v1/HOOK_ENABLED.json').exists()
assert os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize>3*1024**3
# Exercise the exact evaluator wrapper with only its external evaluator call mocked,
# under a diagnostic-only path; concurrent collection-style peer shares that lock.
D.mkdir();(D/'evaluate_locked.py').write_text(SOURCES['evaluate_locked.py'])
mock="""import sys,json,time,subprocess
from pathlib import Path
from types import SimpleNamespace
D=Path(D_PATH);Q=Path(Q_PATH)
def fake(argv,**kw):
 assert argv==['/data/mlr_ifa/env/bin/python','-B',str(Q/'L005_41/evaluate.py')] and kw['cwd']==Q/'L005_41'
 (D/'MOCK_CALL.json').write_text(json.dumps(dict(argv=argv,cwd=str(kw['cwd']),environment_passed=True,optimizer_calls=0,environments=0)))
 time.sleep(.5);return SimpleNamespace(returncode=0)
subprocess.run=fake
source=(D/'evaluate_locked.py').read_text().replace('Q=E.parent','Q=Path('+repr(str(Q))+')').replace("'/data/mlr_ifa/'",repr(str(D)+'/'))
sys.argv=[str(D/'evaluate_locked.py'),'L005_41'];exec(compile(source,str(D/'evaluate_locked.py'),'exec'),dict(__file__=str(D/'evaluate_locked.py'),__name__='__main__'))
""".replace('D_PATH',repr(str(D))).replace('Q_PATH',repr(str(Q)))
peer="""import fcntl,time,json
from pathlib import Path
D=Path(D_PATH);f=(D/LOCK_NAME).open('a');fcntl.flock(f,fcntl.LOCK_EX);a=time.time();time.sleep(.5);b=time.time();(D/'COLLECTION_MOCK.json').write_text(json.dumps(dict(start=a,end=b,optimizer_calls=0,environments=0)));fcntl.flock(f,fcntl.LOCK_UN)
""".replace('D_PATH',repr(str(D))).replace('LOCK_NAME',repr(Q.name+'_collection.lock'))
ps=[]
for name,source in [('evaluation',mock),('collection',peer)]:
 with (D/(name+'.log')).open('x') as f:ps.append(subprocess.Popen([sys.executable,'-B','-c',source],stdout=f,stderr=subprocess.STDOUT))
assert all(p.wait()==0 for p in ps)
rows=[json.loads(x) for x in (D/'L005_41_EVALUATION_LOCK.jsonl').read_text().splitlines()];assert [x['event'] for x in rows]==['acquired','released'];peer_result=js(D/'COLLECTION_MOCK.json');intervals=sorted([(rows[0]['time'],rows[1]['time']),(peer_result['start'],peer_result['end'])]);assert intervals[0][1]<=intervals[1][0]
validation=dict(status='PASS_ZERO_ENVIRONMENT_EVALUATION_COLLECTION_MUTEX',intervals=intervals,source_sha256=sha(D/'evaluate_locked.py'),original_evaluator_argv_cwd=js(D/'MOCK_CALL.json'),optimizer_calls=0,environments=0,production_signals=0,scope='Actual wrapper with evaluator call mocked; diagnostic lock location only; no scientific rollout')
(D/'VALIDATION.json').write_text(json.dumps(validation,indent=2))
# Recheck the live boundary after diagnostics; no prior coordinator receives a signal here.
for k,x in identities.items():
 actual=proc(x['pid']);assert actual['start']==x['start'] and actual['state'] in ['S','R','D']
assert not (PREV/'L005_43_resume.exit').exists() and not (Q/'L005_41/evaluate.log').exists()
h=js(Q/'L005_43/EPOCHS_S.json');assert 289<=len(h)<390
gate=(Q/'L005_43/GATES.jsonl').read_bytes();gate=gate[:gate.rfind(b'\n')+1]
completed={}
for case in ['L005_41','L005_42']:
 assert (Q/case/'train_S.exit').read_text()=='0' and len(js(Q/case/'EPOCHS_S.json'))==400
 completed.update({str(p):sha(p) for p in (Q/case).glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']})
 for stage in range(6,10):
  p=Path('/data/mlr_ifa/experiments')/Q.name/case/'S'/('stage%d.pth'%stage);completed[str(p)]=sha(p)
g=dict(identities,epoch_count=len(h),epoch_prefix_sha256=hashlib.sha256(json.dumps(h,sort_keys=True,separators=(',',':')).encode()).hexdigest(),gate_prefix_bytes=len(gate),gate_prefix_sha256=hashlib.sha256(gate).hexdigest(),completed_hashes=completed,user_authorization='2026-11-02 run one L020 seed alongside current L005 seed; do not interrupt current training',existing_snapshot_contract=sha(PREV/'CONTRACT.json'))
E.mkdir()
for n,s in SOURCES.items():(E/n).write_text(s)
(E/'GUARD.json').write_text(json.dumps(g,indent=2));(E/'ENVIRONMENT_LOCK_VALIDATION.json').write_text(json.dumps(validation,indent=2))
c=dict(version=E.name,files={str(E/n):sha(E/n) for n in [*SOURCES,'GUARD.json','ENVIRONMENT_LOCK_VALIDATION.json']},scientific_contract_sha256=sha(Q/'CONTRACT.json'),prior_engineering_contract=sha(PREV/'CONTRACT.json'),max_training_concurrency=2,early_L020_case='L020_41',no_additional_seeds=True,additional_optimizer_calls=0,additional_environments=0,training_signals=0,completion_source=E.name+'/controller.exit',change='Cross-lambda execution order only; independent ports; original evaluator under collection mutex; original training wrappers unchanged')
(E/'CONTRACT.json').write_text(json.dumps(c,indent=2))
with (E/'controller.log').open('x') as f:p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
r=dict(status='CROSS_LAMBDA_DISPATCHED_PENDING_ACTUAL_ADOPTION',pid=p.pid,time=time.time(),contract_sha256=sha(E/'CONTRACT.json'),contract=c,guard=g,environment_lock_validation=validation)
(E/'DISPATCH.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
