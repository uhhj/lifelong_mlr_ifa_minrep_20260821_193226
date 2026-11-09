import os,json,hashlib,time,stat
from pathlib import Path
os.nice(10)
B=Path('/data/mlr_ifa');target=B/'experiments/TACRST_OBJECT_seed41_20261023/ra_coreset/recovery/latest.pth.tmp';run=target.parents[2]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
files=[]
for root in [B/'experiments',B/'experiment_workspaces',B/'analysis_workspaces']:
 for dp,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if not Path(dp,n).is_symlink()]
  for n in names:
   p=Path(dp,n)
   try:
    st=p.lstat()
    if stat.S_ISREG(st.st_mode) and (n.endswith(('.tmp','.part','.partial','.temp')) or (p.suffix in ['.pth','.pt'] and any(v in n.lower() for v in ['roundtrip','scratch','temporary','serialize']))):files.append(dict(path=str(p),bytes=st.st_size,allocated=st.st_blocks*512,mtime=st.st_mtime))
   except OSError:pass
metadata={}
for p in run.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size<150000 and p.suffix in ['.json','.exit','.log'] and 'buffer' not in p.parts and 'features' not in p.parts:
  if p.name in ['FAILURE.json','TRAINING_COMPLETE.json','RUN_STATUS.json','CONTRACT.json','GPU_PREFLIGHT.json'] or p.suffix=='.exit' or p.name.endswith('.log'):
   metadata[str(p)]=dict(sha256=sha(p),text=p.read_text(errors='replace')[-12000:])
refs=[];contracts=[]
for root in [B/'experiment_workspaces',B/'diagnostics',B/'analysis_workspaces']:
 for dp,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if not Path(dp,n).is_symlink()]
  for n in names:
   p=Path(dp,n)
   if not p.is_file() or p.suffix not in ['.py','.json','.md','.sh','.yaml']:continue
   if 'CONTRACT' in n:contracts.append(dict(path=str(p),sha256=sha(p)))
   if p.stat().st_size>8*1024**2:continue
   text=p.read_text(errors='replace')
   if str(target) in text or 'TACRST_OBJECT_seed41_20261023' in text:
    lines=[v for v in text.splitlines() if str(target) in v or 'TACRST_OBJECT_seed41_20261023' in v];refs.append(dict(path=str(p),sha256=sha(p),lines=[v[:900] for v in lines[:12]]))
open_refs=[];run_procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().decode(errors='replace')
  if str(run) in cmd and int(p.name)!=os.getpid():run_procs.append(int(p.name))
  for fd in (p/'fd').iterdir():
   try:
    linked=os.readlink(fd)
    if str(run) in linked:open_refs.append(dict(pid=int(p.name),target=linked))
   except OSError:pass
 except OSError:pass
siblings=[]
for p in target.parent.iterdir():
 if p.is_file():s=p.stat();siblings.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,sha256=sha(p)))
print(json.dumps(dict(status='READONLY_TEMP_DEPENDENCY_AUDIT',observed=time.time(),temp_candidates=files,target_metadata=metadata,references=refs,contracts=contracts,open_refs=open_refs,run_processes=run_procs,recovery_siblings=siblings,deleted=0)))
