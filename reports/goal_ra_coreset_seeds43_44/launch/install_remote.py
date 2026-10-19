"""Install new directories only, verify code bytes, then one detached finite queue."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,shutil,subprocess,os,time
Q=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019')
B=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018')
A=Path('/tmp/ra_goal_extension_20261019_reviewed_code.zip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not Q.exists(),'Existing queue: inspect, never install/launch twice'
assert shutil.disk_usage('/root').free>6*1024**3
proc=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(' -m coreset_exp.train ' in x or ' -m coreset_exp.evaluate ' in x for x in proc)
for seed in (43,44):assert not Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_20261019'%seed).exists()
with zipfile.ZipFile(A) as z:
 manifest=json.loads(z.read('CODE_MANIFEST.json'));assert set(z.namelist())==set(manifest)|{'CODE_MANIFEST.json'}
 for n,h in manifest.items():
  p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
  assert p.suffix in {'.py','.sh','.md','.json','.yaml','.txt','.sbatch','.patch'} or p.name=='LICENSE'
  assert hashlib.sha256(z.read(n)).hexdigest()==h
 Q.mkdir()
 for seed in (43,44):
  W=Q/('seed%d'%seed);W.mkdir();shutil.copytree(B/'source',W/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.git'))
 for n,h in manifest.items():
  dest=Q/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
 (Q/'CODE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
proof=json.loads((Q/'SOURCE_DIFF.json').read_text())
for seed in (43,44):
 S=Q/('seed%d'%seed)/'source'
 for n,h in proof['unchanged'].items():assert sha(S/n)==sha(B/'source'/n)==h,n
 for n,h in proof['changed'].items():assert sha(S/n)==h,n
 # No additional source file was inherited outside the reviewed source manifest.
 expected=set(proof['unchanged'])|set(proof['changed'])
 actual={p.relative_to(S).as_posix() for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
 assert actual=={n for n in expected if Path(n).suffix in ('.py','.yaml','.sh')},actual-set(expected)
orchestration={n:sha(Q/n) for n in manifest if not '/source/' in n}
(Q/'ORCHESTRATION_SHA256.json').write_text(json.dumps(orchestration,indent=2)+'\n')
(Q/'DEPLOYMENT.json').write_text(json.dumps(dict(status='INSTALLED',time=time.time(),code_archive_sha256=sha(A),unchanged_core=True,files=manifest),indent=2)+'\n')
with (Q/'launcher.log').open('xb') as f:
 p=subprocess.Popen(['bash',str(Q/'supervise_queue.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=str(Q),start_new_session=True)
(Q/'LAUNCH.json').write_text(json.dumps(dict(status='PARTIAL_ADMISSION',time=time.time(),pid=p.pid,command=['bash',str(Q/'supervise_queue.sh')]),indent=2)+'\n')
time.sleep(2);assert p.poll() is None
print(json.dumps(dict(status='LAUNCHED_PARTIAL_ADMISSION',pid=p.pid,queue=str(Q))))
