from pathlib import Path
import json,hashlib,zipfile,time
B=Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023');D=Path('/data/mlr_ifa/experiments/TACRST_OBJECT_seed41_20261023/ra_coreset/recovery')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={}
for rel in ['USER_TERMINATION_BY_USER.json','QUEUE.json','TACRST_41/ENTRY.json']:
 p=B/rel;out[rel]=dict(sha256=sha(p),text=p.read_text())
source={}
for p in (B/'TACRST_41').rglob('*.py'):
 if p.name in ['common.py','train.py','run.py','entry.py','observe.py']:
  s=p.read_text();lines=s.splitlines();relevant=[]
  for i,v in enumerate(lines):
   if 'def save(' in v or 'recovery/latest' in v or 'def checkpoint(' in v or 'os.replace' in v:relevant.append('\n'.join(lines[max(0,i-2):min(len(lines),i+12)]))
  if relevant:source[str(p)]=dict(sha256=sha(p),excerpts=relevant)
print(json.dumps(dict(status='READONLY_STALE_ATOMIC_SAVE_CHECK',observed=time.time(),metadata=out,source=source,partial_is_zip=zipfile.is_zipfile(D/'latest.pth.tmp'),committed_is_zip=zipfile.is_zipfile(D/'latest.pth'),committed_zip_crc_error=zipfile.ZipFile(D/'latest.pth').testzip())))
