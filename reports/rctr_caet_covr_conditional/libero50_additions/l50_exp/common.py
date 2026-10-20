import copy
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

W=Path(os.environ["LIBERO50_WORKSPACE"])
O=Path(os.environ["LIBERO50_OUT"])
S=Path(__file__).resolve().parents[1]
SEED=int(os.environ["LIBERO50_SEED"])
assert SEED in (41,42,43)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("w") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("wb") as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r["python"]);np.random.set_state(r["numpy"])
    torch.set_rng_state(r["cpu"]);torch.cuda.set_rng_state_all(r["cuda"])
def capture(model,optimizer,scheduler):
    return dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
        scheduler=cpu(scheduler.state_dict()),grads={n:cpu(p.grad) for n,p in model.named_parameters()},
        modes={n:m.training for n,m in model.named_modules()},rng=cpu(rng()))
def restore(model,optimizer,scheduler,r):
    model.load_state_dict(r["model"],strict=True);optimizer.load_state_dict(r["optimizer"])
    scheduler.load_state_dict(r["scheduler"])
    for n,p in model.named_parameters():p.grad=None if r["grads"][n] is None else r["grads"][n].to(p.device)
    for n,m in model.named_modules():m.training=r["modes"][n]
    restore_rng(r["rng"])
def verify():
    c=json.loads((O/"CONTRACT.json").read_text())
    actual={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob("*"))
            if p.is_file() and p.suffix in (".py",".yaml",".yml",".sh") and "__pycache__" not in p.parts}
    assert actual==c["source_hashes"]
    for p,h in c["files"].items():assert sha(p)==h,p
    return c
