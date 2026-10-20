"""Four isolated updates and 28 bounded native restoration engineering steps."""
import copy
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O, SEED, verify, write, save
from coreset_exp.model import cfg
from coreset_exp.train import build, PaperMLR, capture, restore
from libero.lifelong.algos.mlr import build_dataset_for_task
from .observe import COVRMLR
from .loss import objective
from .online_admission import run as online_admission


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,"shape"):return __import__("numpy").array_equal(a,b)
    return a==b


def main():
    verify()
    assert not (O/"GPU_PREFLIGHT.json").exists()
    online = online_admission()
    dist.init_process_group("nccl");torch.cuda.set_device(0)
    try:
        c=cfg("ra_coreset")
        c.run_root=str(O/"preflight/smoke")
        c.buffer_dir=str(O/"preflight/smoke/buffer")
        c.experiment_dir=str(O/"preflight/smoke/models")
        a,_=build(c)
        ds=build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6)
        loader=DataLoader(ds,batch_size=10,num_workers=0)
        a.start_task(-1,loader)
        a.refs={6:ds[0][0]["feature"][0,0,0].cuda()}
        a.covr_records=[];a.covr_epoch_statistics=[]
        batch=next(iter(loader))
        before=capture(a)
        one=PaperMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        native=capture(a)
        restore(a,before)
        two=COVRMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        first=capture(a)
        assert one==two and equal(native,first)
        save(O/"preflight/step1.pth",first)
        COVRMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        second=capture(a)
        restore(a,torch.load(O/"preflight/step1.pth",map_location="cpu"))
        COVRMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        assert equal(second,capture(a))
        a.optimizer.zero_grad()
        synthetic=[dict(feature=ds[i][0]["feature"][0],
                        action=ds[i][0]["actions"][0,-1],
                        weight=1. if i==0 else .25) for i in range(4)]
        with torch.random.fork_rng(devices=[0]):
            loss,parts=objective(a.policy.module,synthetic,torch.device("cuda"))
            loss.backward()
        grads=[p.grad for p in a.policy.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0
    finally:
        dist.destroy_process_group()
    verify()
    write(O/"GPU_PREFLIGHT.json",dict(status="PASS",optimizer_calls=4,
        formal_steps=0,environment_steps=28,online_admission=online,zero_aux_native_update_exact=True,
        full_state_epoch_recovery_exact=True,synthetic_gmm_gradient_finite=True,
        synthetic_is_not_recovery_evidence=True,
        real_snapshot_gate="Required before each selected state branches",
        real_certified_gradient="Required before each stage activates COVR",
        gpu=torch.cuda.get_device_name(),torch=torch.__version__))


if __name__=="__main__":
    main()
