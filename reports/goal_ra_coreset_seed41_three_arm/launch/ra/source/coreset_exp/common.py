import hashlib,json,os,random,copy
from pathlib import Path
import numpy as np
import torch
S=Path(__file__).resolve().parents[1]
O=Path(os.environ['CORESET_OUT'])
SEED=int(os.environ['CORESET_SEED'])
assert SEED == 41, 'Only explicitly authorized extension seeds'
SELECTOR_SEED=42 # Frozen calibration/reference subset setting, not training RNG
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def jsonable(x):
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def hashes(root,pattern='*.pt'):return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob(pattern))}
def source_hashes():return {str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
def verify():
    c=json.loads((O/'CONTRACT.json').read_text())
    assert source_hashes()==c['source_sha256'],'Frozen source mismatch'
    for p,h in c['files'].items():assert sha(p)==h,p
    b=json.loads((O/'PARENT_BINDING.json').read_text());P=Path(b['parent'])
    assert sha(P/'CONTRACT.json')==b['contract_sha256']
    assert sha(O/'MODAL_CONTRACT.json')==sha(P/'MODAL_CONTRACT.json')==b['modal_sha256']
    assert sha(O/'FEATURES.json')==sha(P/'FEATURES.json')==b['feature_manifest_sha256']
    assert hashes(O/'features')==json.loads((O/'FEATURES.json').read_text())['hashes']
    return c
