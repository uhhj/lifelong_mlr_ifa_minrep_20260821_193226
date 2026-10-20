"""Four isolated actual raw-image GPU updates; no environment or formal training."""
import copy
import json
import time
import torch
import torch.distributed as dist
from .common import O,verify,write,capture,restore
from .data import cfg
from .pretrain import build,loss

def main():
    verify();assert not (O/"PRETRAIN_GPU_PREFLIGHT.json").exists()
    assert torch.cuda.is_available()
    dist.init_process_group("nccl");torch.cuda.set_device(0)
    try:
        c=cfg("pretrain");model,opt,sched,ds,sampler,loader,names=build(c)
        sampler.set_epoch(0);it=iter(loader);batches=[next(it),next(it)]
        before=capture(model,opt,sched)
        fixed={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        def two():
            values=[];begin=time.monotonic()
            for data in batches:
                model.train();opt.zero_grad();value=loss(model,copy.deepcopy(data))
                assert torch.isfinite(value)
                (c.train.loss_scale*value).backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.named_parameters() if ".lora_" in n)
                if c.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip)
                opt.step();sched.step();values.append(float(value.detach()))
            torch.cuda.synchronize()
            return values,time.monotonic()-begin,capture(model,opt,sched)
        first,seconds,after=two()
        restore(model,opt,sched,before)
        second,seconds2,replayed=two()
        assert first==second
        assert all(torch.equal(after["model"][k],replayed["model"][k]) for k in after["model"])
        assert all(torch.equal(p.detach().cpu(),fixed[n]) for n,p in model.named_parameters() if n in fixed)
        changed=[n for n,p in model.named_parameters() if not torch.equal(p.detach().cpu(),before["model"][n])]
        assert any(".lora_" in n for n in changed)
        assert any("state_encoder." in n for n in changed)
        write(O/"PRETRAIN_GPU_PREFLIGHT.json",dict(status="PASS",optimizer_calls=4,
            environment_steps=0,formal_updates=0,batch_size=10,fp32=True,
            frozen_clip_parameters_unchanged=True,resume_tensor_exact=True,
            changed_parameter_count=len(changed),trainable_parameter_count=len(names),
            measured_two_updates_seconds=[seconds,seconds2],expected_steps=100*len(loader),
            approximate_seconds=100*len(loader)*(seconds+seconds2)/4,
            peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    finally:dist.destroy_process_group()
if __name__=="__main__":main()
