"""Disclosed shared 25-task pretraining reproduction, not authors' released weights."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
import robomimic.utils.tensor_utils as TensorUtils
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.utils import control_seed,make_linear_schedule_with_warmup
from .common import O,sha,verify,write,save,capture,restore,cpu
from .data import cfg,pretrain_dataset

def loss(model,data):
    p=model.module if isinstance(model,DDP) else model
    data=TensorUtils.map_tensor(data,lambda x:x.cuda(non_blocking=True))
    data=p.preprocess_input(data)
    # Spatial representation and token readout are exactly the downstream architecture.
    x=p.spatial_encode(data)
    last=p.temporal_encode(x)[:,-1]
    return p.policy_head.loss_fn(p.policy_head(last),data["actions"])

def build(c):
    assert c.pretraining_mode is True and c.seed==42 and c.pretrain is None
    control_seed(42)
    dataset,shape=pretrain_dataset(c);c.shape_meta=shape
    sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True,seed=0)
    loader=DataLoader(dataset,batch_size=10,sampler=sampler,num_workers=c.train.num_workers,
                      persistent_workers=False)
    model=CompletePolicy(c,shape).cuda()
    allowed=("image_proj.","film_scale.","film_shift.","state_encoder.","state_scale.",
             "state_shift.","temporal_transformer.","policy_head.")
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all(".lora_" in n or n.startswith(allowed) for n in names)
    assert any(n.startswith("state_encoder.") for n in names)
    assert any(n.startswith("image_encoders.") and ".lora_" in n for n in names)
    assert any(n.startswith("language_encoder.") and ".lora_" in n for n in names)
    model=DDP(model,device_ids=[0],output_device=0)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               **c.train.optimizer.kwargs)
    assert optimizer.param_groups[0]["lr"]==1e-4
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        make_linear_schedule_with_warmup(500,100*len(loader)))
    return model,optimizer,scheduler,dataset,sampler,loader,names

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");a=ap.parse_args()
    verify()
    assert json.loads((O/"PRETRAIN_GPU_PREFLIGHT.json").read_text())["status"]=="PASS"
    assert not (O/"PRETRAIN_COMPLETE.json").exists()
    checkpoint=O/"pretraining/recovery.pth"
    assert a.resume or not checkpoint.exists()
    dist.init_process_group("nccl");torch.cuda.set_device(0)
    try:
        c=cfg("pretrain");model,opt,sched,ds,sampler,loader,names=build(c)
        start=0;steps=0;history=[]
        if a.resume:
            old=torch.load(checkpoint,map_location="cpu")
            assert old["contract_sha256"]==sha(O/"CONTRACT.json")
            start,steps,history=old["next_epoch"],old["steps"],old["history"]
            restore(model,opt,sched,old["state"])
        else:
            write(O/"PRETRAIN_INITIALIZATION.json",dict(seed=42,trainable_names=names,
                windows=len(ds),demos=ds.n_demos,batches=len(loader),epochs=100,
                expected_steps=100*len(loader),source="new shared pretraining reproduction",
                official_task_checkpoint_loaded=False))
        for epoch in range(start,100):
            sampler.set_epoch(epoch);model.train();begin=time.monotonic();total=0.
            for data in loader:
                opt.zero_grad();value=loss(model,data);assert torch.isfinite(value)
                (c.train.loss_scale*value).backward()
                if c.train.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip)
                opt.step();sched.step();steps+=1;total+=float(value.detach())
            history.append(dict(epoch=epoch,steps=steps,batches=len(loader),
                                loss=total/len(loader),seconds=time.monotonic()-begin))
            save(checkpoint,dict(state=capture(model,opt,sched),next_epoch=epoch+1,
                steps=steps,history=history,contract_sha256=sha(O/"CONTRACT.json")))
            write(O/"pretraining/epochs.json",history)
            write(O/"RUN_STATUS.json",dict(status="PRETRAINING",epoch=epoch,steps=steps,
                                          pid=os.getpid(),unix=time.time()))
            print(json.dumps(history[-1]),flush=True)
        assert len(history)==100 and steps==100*len(loader)
        write(O/"pretraining/epochs.json",history)
        final=O/"pretraining/libero50_reproduced_multitask_ep99.pth"
        c.pretraining_mode=False
        save(final,dict(state_dict=cpu(model.module.state_dict()),cfg=c,epoch=99,
            source="reproduced; not an official LIBERO50 checkpoint",
            contract_sha256=sha(O/"CONTRACT.json")))
        write(O/"PRETRAIN_COMPLETE.json",dict(status="TRAINED_AWAITING_STRICT_LOAD",
            epochs=100,steps=steps,windows=len(ds),demos=ds.n_demos,
            final_checkpoint=str(final),sha256=sha(final),
            summed_epoch_seconds=sum(x["seconds"] for x in history),
            no_closed_loop_selection=True))
    finally:dist.destroy_process_group()

if __name__=="__main__":main()
