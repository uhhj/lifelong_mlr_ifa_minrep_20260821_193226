import os,json,hashlib,pickle
from pathlib import Path
import numpy as np
import torch
from coreset_exp.common import sha,write,save,rng,restore_rng,hashes,jsonable
from coreset_exp.train import cpu
O=Path(os.environ['ZR_RUN']);S=Path(__file__).resolve().parents[1]
P=Path(os.environ['PARENT_RUN']);PROTO=Path(os.environ['PROTOTYPE_RUN'])
SEED=int(os.environ['ZR_SEED']);assert SEED in (40,41)
ARM='coreset_ifa_zr';ROOT=O/ARM
def sources():
    return {str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
def digest(sd):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in sd.items()}
def rng_digest(r=None):
    r=rng() if r is None else r
    return {k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in r.items()}
def numeric():return dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,algorithms=torch.are_deterministic_algorithms_enabled(),cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
def assert_parent_numeric(evaluation=False):
    expected=json.loads((S.parent/'NUMERIC_ADMISSION.json').read_text())['numeric'];actual=numeric()
    for key in ('torch','cuda','gpu','matmul_tf32','cudnn_tf32','deterministic','algorithms'):assert actual[key]==expected[key],(key,actual[key],expected[key])
    assert actual['benchmark']==evaluation and actual['cublas'] is None
    return actual
def append(path,row):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def verify(frozen=True):
    inputs=json.loads((O/'INPUT_SHA256.json').read_text())
    for p,h in inputs.items():assert sha(p)==h,('Input changed',p)
    assert sha(O/'MODAL_CONTRACT.json')==sha(P/'MODAL_CONTRACT.json')
    if frozen:
        c=json.loads((O/'CONTRACT.json').read_text());assert sources()==c['source_sha256']
        for p,h in c['configs'].items():assert sha(p)==h,p
    return True
