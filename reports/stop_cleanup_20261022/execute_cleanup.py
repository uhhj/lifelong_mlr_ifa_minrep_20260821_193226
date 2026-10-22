"""Exact reviewed disposable-file cleanup; no recursive experiment deletion."""
import os,json,hashlib,time,subprocess,stat
from pathlib import Path
R=Path('/data/mlr_ifa/cleanup_20261022')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(R/'PLAN.json')=='34afd3ec12ebc9055058a9b4fc78dafc6ec74148d7125583fd89bafbe3a4b688'
d=json.loads((R/'PLAN.json').read_text()); selected=d['delete_files']
assert len(selected)==37 and not (R/'CLEANUP_RECEIPT.json').exists()
roots=[Path('/data/mlr_ifa/experiments'),Path('/root/mlr_ifa/experiments')]
targets={x['path'] for x in selected}
def quiet():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for q in Path('/proc').iterdir():
  if not q.name.isdigit() or int(q.name)==os.getpid():continue
  try:
   cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   cwd=os.readlink(str(q/'cwd'))
  except (OSError,ProcessLookupError):continue
  assert '/experiment_workspaces/' not in cwd and '/experiment_workspaces/' not in cmd,(q.name,cmd)
  for fd in (q/'fd').glob('*'):
   try: dest=os.readlink(str(fd))
   except OSError:continue
   assert dest not in targets,(q.name,dest)
quiet()
evidence={e['path']:e['sha256'] for x in selected+d['skipped'] for e in x['evidence']}
for p,h in evidence.items():assert sha(p)==h,p
for x in selected:
 p=Path(x['path']);s=p.lstat()
 assert p.resolve()==p and any(r in p.parents for r in roots)
 assert 'smoke' in p.parts or 'preflight' in p.parts
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==x['bytes'] and s.st_mtime==x['mtime']
 assert not x['contract_references'] and any(e['status']=='PASS' for e in x['evidence'])
 assert sha(p)==x['sha256'],str(p)
for x in d['skipped']:assert sha(x['path'])==x['sha256']
def disk():
 return {p:dict(ordinary_free=(s:=os.statvfs(p)).f_bavail*s.f_frsize,root_free=s.f_bfree*s.f_frsize) for p in ['/data','/']}
before=disk();quiet()
(R/'EXECUTION_PLAN.json').write_text(json.dumps(dict(unix=time.time(),plan_sha256=sha(R/'PLAN.json'),files=selected),indent=2)+'\n')
deleted=[]
with (R/'DELETION_LOG.jsonl').open('x') as log:
 for x in selected:
  p=Path(x['path']);s=p.lstat()
  assert s.st_size==x['bytes'] and s.st_mtime==x['mtime'] and s.st_nlink==1 and p.resolve()==p
  p.unlink()
  row=dict(path=str(p),sha256=x['sha256'],bytes=x['bytes'],allocated=x['allocated'],unix=time.time())
  deleted.append(row);log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
os.sync()
after=disk()
assert all(not Path(x['path']).exists() for x in selected)
for p,h in evidence.items():assert sha(p)==h,p
for x in d['skipped']:assert sha(x['path'])==x['sha256']
quiet()
receipt=dict(status='CLEANUP_COMPLETE',unix=time.time(),deleted=deleted,skipped_preserved=len(d['skipped']),evidence_unchanged=len(evidence),before=before,after=after,released={p:after[p]['root_free']-before[p]['root_free'] for p in before},gpu_empty=True,experiment_processes_absent=True,scope='37 isolated smoke/preflight serialization artifacts only; formal checkpoints, actual recovery, data, features and buffers retained')
(R/'CLEANUP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
