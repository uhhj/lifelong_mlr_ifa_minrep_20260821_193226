import os,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/gn_functional_geometry_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1024*1024),b''):h.update(z)
 return h.hexdigest()
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(z,indent=2,allow_nan=False,ensure_ascii=False)+'\n');os.replace(t,p)
def table(name,rows):
 p=OUT/name;t=p.with_name(p.name+'.tmp');keys=list(dict.fromkeys(k for r in rows for k in r))
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def save(p,**z):
 import numpy as np
 p=Path(p);t=p.with_name(p.name+'.tmp.npz');np.savez(t,**z);os.replace(t,p)
def read(p):return json.loads(Path(p).read_text())
