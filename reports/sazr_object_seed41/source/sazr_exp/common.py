import os,sys,json,time,hashlib,copy
from pathlib import Path
W=Path(__file__).resolve().parents[2]
P=Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019")
S=Path("/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed41/source")
O=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
CP=P/"ra_coreset/models/mlrifa_6_model_ep99.pth"
os.environ.update(CORESET_OUT=str(P),CORESET_SEED="41",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
sys.path.append(str(S))
import numpy as np
import torch
from coreset_exp.common import sha,write,save,rng,restore_rng
from coreset_exp.model import cfg,strict_downstream,complete,data_setup
from libero.lifelong.utils import control_seed,make_linear_schedule_with_warmup
from libero.lifelong.models.brief import BriefPolicy
from .core import Residual,corrected,basis,noise_table,choose_anchor_times
def js(p):return json.loads(Path(p).read_text())
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as f:f.write(json.dumps(x,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def status(s,**kw):write(O/"RUN_STATUS.json",dict(status=s,time=time.time(),pid=os.getpid(),**kw))
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
    return copy.deepcopy(x)
def tensor_hash(m):
    h=hashlib.sha256()
    for n,v in sorted(m.state_dict().items()):
        a=v.detach().cpu().contiguous();h.update(n.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def setup():
    assert torch.cuda.is_available()
    torch.set_num_threads(4);control_seed(41)
    f=js("/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3/ADMISSION.json")["numeric_flags"]
    torch.backends.cuda.matmul.allow_tf32=f["tf32_matmul"];torch.backends.cudnn.allow_tf32=f["tf32_cudnn"]
    torch.backends.cudnn.benchmark=f["cudnn_benchmark"];torch.use_deterministic_algorithms(f["deterministic"])
def verify():
    c=js(O/"CONTRACT.json")
    for p,h in c["files"].items():assert sha(p)==h,("Frozen input changed",p)
    return c
def new_public():
    c=cfg("ra_coreset");c.aug=False;c.train.use_augmentation=False
    p=BriefPolicy(c,None).cuda().eval().requires_grad_(False);strict_downstream(p,CP)
    return p
def make_bank(q):
    d=np.load(W/"language.npz")
    return Residual(q,torch.from_numpy(d["canonical"]),torch.from_numpy(d["gate"])).cuda()
def raw(head,z):
    assert not head.low_eval_noise
    x=head.share(z)
    return head.mean_layer(x).view(-1,5,7),head.actv(head.logstd_layer(x).view(-1,5,7))+head.min_std,head.logits_layer(x)
def decode(p,bank,h):
    with torch.no_grad():
        z=p.temporal_encode(h)[:,-1];b,s,l=raw(p.policy_head,z)
    e=h[:,-1,0];assert bank.route(e).all()
    delta=bank(z,e);out=corrected(b,s,l,delta)
    out.update(z=z,b=b,s=s,l=l,delta=delta)
    return out
def optimizer(bank):
    opt=torch.optim.AdamW(bank.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.1)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,make_linear_schedule_with_warmup(500,1570))
    return opt,sch
def update(p,bank,opt,sch,h,a):
    opt.zero_grad(set_to_none=True);out=decode(p,bank,h)
    loss=(out["dist"].base_dist.loc-a[:,-1]).abs().mean();assert torch.isfinite(loss)
    loss.backward();assert all(v.grad is None for v in p.parameters())
    norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),100.);assert torch.isfinite(norm)
    opt.step();sch.step()
    return dict(loss=float(loss),grad_norm=float(norm),delta_max=float(out["delta"].abs().max()))
