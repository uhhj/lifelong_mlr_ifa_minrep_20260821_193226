import json,os,hashlib,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/functional_component_counterfactual_v1'
SCRATCH=Path('/root/functional_component_counterfactual_v1')
SOURCE=ROOT/'reports/actual_update_fidelity_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n');os.replace(t,p)
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def save(p,**x):
 import numpy as np
 p=Path(p);t=p.with_name(p.name+'.tmp.npz');np.savez(t,**x);os.replace(t,p)
def status(stage,**kw):
 z=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);z["formal_training_resumed"]=False
 write(OUT/'STATUS.json',z);print(stage,kw,flush=True)
def stable_seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
METHODS=['GN','C1_0','C1_1','C1_2','C2_0','C2_1','C2_2','C3']
def label(old,new):
 if old=='NEG' and new=='NEG':return 'JOINT_NLL_BENEFIT'
 if old=='POS' and new=='NEG':return 'OLD_NEW_NLL_TRADEOFF'
 if old=='NEG' and new=='POS':return 'OLD_FAVORED'
 if old=='POS' and new=='POS':return 'BOTH_WORSE'
 if old=='NEUTRAL' and new=='NEG':return 'NEW_BENEFIT_OLD_NEUTRAL'
 return 'UNRESOLVED'
