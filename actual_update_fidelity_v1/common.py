import csv, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/actual_update_fidelity_v1'
SCRATCH=Path('/root/actual_update_fidelity_v1')
ALPHAS=[1.,.5,.25,.125,.0625,.03125]
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [],lineterminator='\n');w.writeheader();w.writerows(rows)
 os.replace(t,p)
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def emit(p,v):
 with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()
