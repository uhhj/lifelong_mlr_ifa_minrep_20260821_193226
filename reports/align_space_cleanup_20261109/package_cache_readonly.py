import os,json,time,hashlib,stat
from pathlib import Path
C=Path('/data/mlr_ifa/cache');E=Path('/data/mlr_ifa/env');candidates=[];kept={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in sorted((E/'conda-meta').glob('*.json')):kept[str(p)]=sha(p)
for p in sorted((C/'conda').glob('*.conda')):
 s=p.lstat();record=E/'conda-meta'/(p.stem+'.json');r=json.loads(record.read_text()) if record.exists() else None
 assert not p.is_symlink() and stat.S_ISREG(s.st_mode)
 actual=sha(p)
 if r and r.get('sha256'):assert actual==r['sha256']
 candidates.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,sha256=actual,nlink=s.st_nlink,installed_record=str(record) if r else None,record_archive_sha_matched=bool(r and r.get('sha256')==actual)))
for p in (C/'pip/http-v2').rglob('*'):
 if p.is_file() and not p.is_symlink():
  s=p.stat();candidates.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,sha256=sha(p),nlink=s.st_nlink,kind='pip_http_download_cache'))
refs=[];active_installers=[];open_refs=[]
targets={r['path'] for r in candidates}
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and any(v in cmd for v in ['pip install','pip download','conda install','conda create','mamba install','mamba create']):active_installers.append(dict(pid=int(p.name),cmd=cmd))
  for fd in (p/'fd').iterdir():
   try:
    s=os.readlink(fd)
    if s in targets:open_refs.append(dict(pid=int(p.name),path=s))
   except OSError:pass
 except OSError:pass
for root in ['/data/mlr_ifa/experiment_workspaces','/data/mlr_ifa/diagnostics']:
 for p in Path(root).rglob('*CONTRACT*.json'):
  if p.is_file():
   text=p.read_text()
   for t in targets:
    if t in text:refs.append(dict(contract=str(p),target=t))
print(json.dumps(dict(status='READONLY_PACKAGE_DOWNLOAD_CACHE_AUDIT',observed=time.time(),candidates=candidates,bytes_reclaimable=sum(r['allocated'] for r in candidates),environment_records_sha256=kept,contract_references=refs,open_refs=open_refs,active_installers=active_installers,extracted_packages_and_huggingface_excluded=True,deleted=0)))
