import os,json,time,stat
from pathlib import Path
os.nice(10)
roots=['/data/mlr_ifa/diagnostics','/data/mlr_ifa/tmp','/data/tmp','/tmp','/var/tmp','/root/.cache/pip','/data/mlr_ifa/.cache','/data/.cache']
out={}
for name in roots:
 p=Path(name);rows=[];groups={};count=0
 if p.exists():
  for directory,dirs,files in os.walk(p,followlinks=False):
   dirs[:]=[d for d in dirs if not Path(directory,d).is_symlink()]
   for f in files:
    q=Path(directory,f)
    try:
     s=q.lstat()
     if not stat.S_ISREG(s.st_mode):continue
     count+=1;group=q.relative_to(p).parts[0];groups[group]=groups.get(group,0)+s.st_blocks*512
     if s.st_size>=16*1024**2:rows.append(dict(path=str(q),bytes=s.st_size,allocated=s.st_blocks*512,mtime=s.st_mtime,nlink=s.st_nlink))
    except OSError:pass
 out[name]=dict(exists=p.exists(),regular_files=count,allocated=sum(groups.values()),groups=sorted(groups.items(),key=lambda x:-x[1])[:60],large_files=sorted(rows,key=lambda x:-x['allocated'])[:150])
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if '/data/mlr_ifa' in cmd and int(p.name)!=os.getpid():procs.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
print(json.dumps(dict(status='READONLY_TEMP_INVENTORY_NO_DELETION',observed=time.time(),roots=out,processes=procs,free={p:dict(available=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize,free=os.statvfs(p).f_bfree*os.statvfs(p).f_frsize,device=os.stat(p).st_dev) for p in ['/','/data','/tmp','/root/.cache']},deleted=0)))
