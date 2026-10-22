import os,json,subprocess
from pathlib import Path
rows=[]
roots=[Path('/data/mlr_ifa/experiments'),Path('/root/mlr_ifa/experiments'),Path('/data/mlr_ifa/experiment_workspaces'),Path('/data/mlr_ifa/analysis_workspaces')]
for root in roots:
 if not root.exists():continue
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if n not in ('features','buffer','source','.git','latent','demos') and not (Path(base)/n).is_symlink()]
  for name in files:
   p=Path(base)/name
   if p.is_symlink():continue
   st=p.stat()
   if st.st_size>50000000 and any(k in str(p).lower() for k in ('preflight','smoke','probe','admission','tmp')):
    rows.append(dict(path=str(p),bytes=st.st_size,allocated=st.st_blocks*512,links=st.st_nlink,mtime=st.st_mtime))
print(json.dumps(dict(candidates=rows,cache=subprocess.check_output(['du','-x','-B1','--max-depth=2','/data/mlr_ifa/cache'],text=True),tmp=subprocess.check_output(['du','-x','-B1','--max-depth=1','/tmp'],text=True,stderr=subprocess.DEVNULL))))
