"""Package only code and lightweight audit evidence, never model/data payloads."""
import tarfile,json,hashlib,time
from pathlib import Path
Q=Path(__file__).resolve().parent
allowed={'.json','.jsonl','.md','.yaml','.csv','.log','.txt','.py','.sh','.patch','.exit','.pid','.started'}
items={}
for p in Q.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in allowed and not {'__pycache__','assets','monitor'}&set(p.parts):
  if 'prelaunch_failures' not in p.parts and p.name=='PACKAGE_MANIFEST.json':continue
  items['queue/'+p.relative_to(Q).as_posix()]=p
for seed in (42,43,44):
 O=Path(f'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{seed}_20261017')
 for p in O.rglob('*'):
  if p.is_file() and p.suffix in allowed and not {'features','buffer','models','recovery','smoke','monitor'}&set(p.relative_to(O).parts):items[f'seed{seed}/'+p.relative_to(O).as_posix()]=p
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(items.items())}
(Q/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(Q/'LIGHT_EVIDENCE.tar.gz','w:gz') as t:
 for n,p in sorted(items.items()):t.add(p,arcname=n,recursive=False)
 t.add(Q/'PACKAGE_MANIFEST.json',arcname='PACKAGE_MANIFEST.json')
print(json.dumps(dict(files=len(items),bytes=sum(x['bytes'] for x in manifest.values()),archive_bytes=(Q/'LIGHT_EVIDENCE.tar.gz').stat().st_size)))
