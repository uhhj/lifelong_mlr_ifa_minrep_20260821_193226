import os,json,time,stat,collections
from pathlib import Path
os.nice(10)
roots=['/data','/root/.cache','/root/.local/share/Trash','/var/cache/apt/archives'];out={}
for root in roots:
 base=Path(root);groups=collections.defaultdict(int);candidates=[];count=0;device=base.stat().st_dev if base.exists() else None
 for directory,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if not Path(directory,d).is_symlink() and Path(directory,d).stat().st_dev==device]
  for name in files:
   p=Path(directory,name)
   try:
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode):continue
    count+=1;parts=p.relative_to(base).parts;groups['/'.join(parts[:2])]+=s.st_blocks*512
    suffix=name.lower();temp=any(suffix.endswith(x) for x in ['.tmp','.temp','.part','.partial','.bak','.zip','.tar','.tar.gz','.tgz','.whl','.deb','.run','.old'])
    if (s.st_size>=32*1024**2 and temp) or (s.st_size>=128*1024**2 and not any(x in parts for x in ['experiments','experiment_workspaces','datasets','checkpoints','env','src','huggingface'])):
     candidates.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,mtime=s.st_mtime,nlink=s.st_nlink))
   except OSError:pass
 out[root]=dict(count=count,groups=sorted(groups.items(),key=lambda x:-x[1])[:80],potential_files=sorted(candidates,key=lambda x:-x['allocated'])[:180])
print(json.dumps(dict(status='READONLY_EXTENDED_SPACE_INVENTORY',observed=time.time(),roots=out,deleted=0)))
