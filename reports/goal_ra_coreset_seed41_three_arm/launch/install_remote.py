"""Install new namespaces only; preserve parent scientific bytes and symlinks."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,shutil,subprocess,time
Q=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019')
A=Path('/tmp/ra_goal41_reviewed_code.zip')
parents=dict(baseline=Path('/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed42/source'),ra=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019/seed43/source'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not Q.exists(),'Existing queue: inspect, never reinstall'
assert shutil.disk_usage('/root').free>6*1024**3
proc=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(' -m coreset_exp.train ' in x or ' -m coreset_exp.evaluate ' in x for x in proc)
for name in ['Coreset_paper','RA_Coreset']:assert not Path('/root/mlr_ifa/experiments/'+name+'_GOAL_seed41_20261019').exists()
with zipfile.ZipFile(A) as z:
 m=json.loads(z.read('CODE_MANIFEST.json'));assert set(z.namelist())==set(m)|{'CODE_MANIFEST.json'}
 for n,h in m.items():
  p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
  assert p.suffix in {'.py','.sh','.md','.json','.yaml','.txt','.sbatch','.patch'} or p.name=='LICENSE'
  assert hashlib.sha256(z.read(n)).hexdigest()==h
 Q.mkdir()
 for arm,p in parents.items():
  (Q/arm).mkdir();shutil.copytree(p,Q/arm/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.git'))
 for n in m:
  p=Q/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
 (Q/'CODE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
proof=json.loads((Q/'SOURCE_DIFF.json').read_text())
for arm,parent in parents.items():
 S=Q/arm/'source';v=proof[arm]
 for n,h in v['unchanged'].items():assert sha(S/n)==sha(parent/n)==h,n
 for n,h in v['changed'].items():assert sha(S/n)==h,n
 expected={n for n in {**v['unchanged'],**v['changed']} if Path(n).suffix in ('.py','.sh','.yaml')}
 actual={p.relative_to(S).as_posix() for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml') and '__pycache__' not in p.parts}
 assert actual==expected,(arm,actual-expected)
orchestration={str(Q/n):sha(Q/n) for n in m if '/source/' not in n}
(Q/'ORCHESTRATION_SHA256.json').write_text(json.dumps(orchestration,indent=2)+'\n')
(Q/'DEPLOYMENT.json').write_text(json.dumps(dict(status='INSTALLED',time=time.time(),archive_sha256=sha(A),scientific_core_unchanged=True,files=m),indent=2)+'\n')
with (Q/'launcher.log').open('xb') as f:p=subprocess.Popen(['bash',str(Q/'supervise_queue.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=str(Q),start_new_session=True)
(Q/'LAUNCH.json').write_text(json.dumps(dict(status='PARTIAL_ADMISSION',time=time.time(),pid=p.pid,command=['bash',str(Q/'supervise_queue.sh')]),indent=2)+'\n')
time.sleep(2)
print(json.dumps(dict(status='LAUNCHED_PARTIAL_ADMISSION',pid=p.pid,queue=str(Q),immediate_exit=p.poll())))
