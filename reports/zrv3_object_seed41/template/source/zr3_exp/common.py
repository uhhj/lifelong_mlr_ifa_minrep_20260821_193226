import os,sys,json,copy,time,hashlib
from pathlib import Path
from .params import *
os.environ.update(CORESET_OUT=str(P),CORESET_SEED=str(SEED),PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
sys.path[:0]=[str(W/'attachment'),str(S)]
import numpy as np
import torch
from coreset_exp.common import sha,write,save,rng,restore_rng
from coreset_exp.model import cfg,strict_downstream
from libero.lifelong.utils import control_seed,make_linear_schedule_with_warmup
from libero.lifelong.models.brief import BriefPolicy
from reference_core import MeanCalibrationBank,corrected_gmm,extend_gates

def js(p):return json.loads(Path(p).read_text())
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def status(s,**kw):write(O/'RUN_STATUS.json',dict(status=s,time=time.time(),pid=os.getpid(),**kw))
def tensor_hash(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous();h.update(n.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def setup():
    assert torch.cuda.is_available();torch.set_num_threads(4);control_seed(SEED)
    flags=js(O/'ADMISSION.json')['numeric_flags']
    torch.backends.cuda.matmul.allow_tf32=flags['tf32_matmul'];torch.backends.cudnn.allow_tf32=flags['tf32_cudnn'];torch.backends.cudnn.benchmark=flags['cudnn_benchmark']
    torch.use_deterministic_algorithms(flags['deterministic'])
def verify():
    c=js(O/'CONTRACT.json')
    for p,h in c['files'].items():assert sha(p)==h,('Frozen source/input mismatch',p)
    return c
def new_public():
    c=cfg(PARM);c.aug=False;c.train.use_augmentation=False
    p=BriefPolicy(c,None).cuda().eval().requires_grad_(False);strict_downstream(p,c.pretrain)
    t0=copy.deepcopy(p.temporal_transformer).eval().requires_grad_(False)
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(p.temporal_transformer.parameters(),t0.parameters()))
    return p,t0
def load_stage(p,t):
    b=js(O/'PARENT_BINDING.json')['checkpoints'][str(t)];assert sha(b['path'])==b['sha256'];strict_downstream(p,b['path']);p.eval().requires_grad_(False)
    return b
def make_bank(t,previous=None):
    # Previously admitted canonical data; construction is still strictly prefix-only.
    e=np.load(W/'language_and_gates.npz')['canonical'];w=np.empty((0,512));reports=[]
    for k in range(6,t+1):w,r=extend_gates(e[:k],e[k],w);reports.append(r)
    before=cpu(rng());b=MeanCalibrationBank(range(6,t+1),w,experiment_seed=SEED).cuda()
    if previous is not None:
        for n,v in previous.branches.items():
            b.branches[n].load_state_dict(v.state_dict(),strict=True)
            assert all(torch.equal(a,c) for a,c in zip(v.parameters(),b.branches[n].parameters()))
    assert torch.equal(before['cpu'],torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before['cuda'],torch.cuda.get_rng_state_all()))
    return b,reports
def optimizer(bank,total):
    opt=torch.optim.AdamW(bank.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.1)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,make_linear_schedule_with_warmup(500,total))
    return opt,sch
