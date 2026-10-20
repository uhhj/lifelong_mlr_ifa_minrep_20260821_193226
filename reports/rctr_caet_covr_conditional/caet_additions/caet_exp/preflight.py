"""Exactly four isolated native policy updates, zero environment steps."""
import copy
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O, SEED, verify, write, save
from coreset_exp.model import cfg
from coreset_exp.train import build, PaperMLR, capture, restore
from libero.lifelong.algos.mlr import build_dataset_for_task
from .observe import CAETMLR
from .dataset import EffectDataset
from .model import EffectModel


def equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):
        return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,"shape"):
        return __import__("numpy").array_equal(a,b)
    return a==b


def main():
    verify()
    assert not (O/"GPU_PREFLIGHT.json").exists()
    assert json.loads((O/"effects/current/task6/AUDIT.json").read_text())["status"]=="PASS"
    dist.init_process_group("nccl");torch.cuda.set_device(0)
    try:
        c=cfg("ra_coreset")
        c.run_root=str(O/"preflight/smoke")
        c.buffer_dir=str(O/"preflight/smoke/buffer")
        c.experiment_dir=str(O/"preflight/smoke/models")
        a,_=build(c)
        native=build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6)
        labeled=EffectDataset(native,6)
        assert native.file_paths==labeled.file_paths
        loader=DataLoader(labeled,batch_size=10,num_workers=0)
        a.start_task(-1,loader)
        a.refs={6:native[0][0]["feature"][0,0,0].cuda()}
        a.effect=None;a.caet_epoch_statistics=[]
        batch=next(iter(loader))
        initial=capture(a)
        one=PaperMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        after_native=capture(a)
        restore(a,initial)
        two=CAETMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        after_new=capture(a)
        assert one==two and equal(after_native,after_new)
        save(O/"preflight/serialized_step1.pth",after_new)
        CAETMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        after_step2=capture(a)
        restore(a,torch.load(O/"preflight/serialized_step1.pth",map_location="cpu"))
        CAETMLR.observe(a,copy.deepcopy(batch),6,0,{"selected_pairs":[]})
        assert equal(after_step2,capture(a))
        # Interface differentiability only; fitted, held-out physical prediction
        # and real cross-task-pair checks are mandatory before Task7 activation.
        with torch.random.fork_rng(devices=[0]):
            model=EffectModel().cuda().eval().requires_grad_(False)
            x=batch[0]["feature"].cuda()
            actions=batch[0]["actions"][:,-1] if batch[0]["actions"].ndim==3 else batch[0]["actions"][:,0,-1]
            actions=actions.cuda().detach().requires_grad_(True)
            output,_=model(x,actions)
            output.square().mean().backward()
            assert torch.isfinite(actions.grad).all() and float(actions.grad.abs().sum())>0
            assert all(p.grad is None for p in model.parameters())
    finally:
        dist.destroy_process_group()
    verify()
    write(O/"GPU_PREFLIGHT.json",dict(status="PASS",optimizer_calls=4,
        formal_steps=0,environment_steps=0,zero_aux_native_update_exact=True,
        full_state_epoch_recovery_exact=True,actual_index_wrapper_order_exact=True,
        frozen_effect_action_gradient_finite_nonzero=True,
        fitted_predictor_validation="Required per stage before Task7-9 activation",
        gpu=torch.cuda.get_device_name(),torch=torch.__version__))


if __name__=="__main__":
    main()
