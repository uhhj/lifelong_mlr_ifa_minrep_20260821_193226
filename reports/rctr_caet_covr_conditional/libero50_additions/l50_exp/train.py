"""Five simultaneous new tasks per stage, unchanged main optimizer-step budget."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.algos.mlr import FileTrajectoryDataset
from libero.lifelong.utils import control_seed,make_linear_schedule_with_warmup,torch_load_model
from .common import O,SEED,verify,sha,write,save,cpu,capture,restore,rng,restore_rng
from .data import cfg
from .storage import store,buffer_hashes
from . import grouped_ifa

def build(c):
    control_seed(SEED)
    p=BriefPolicy(c,c.shape_meta).cuda()
    source=torch_load_model(c.pretrain)[0];wanted=p.state_dict()
    subset={k:v for k,v in source.items() if k in wanted}
    assert set(subset)==set(wanted)
    p.load_state_dict(subset,strict=True)
    assert all(torch.equal(p.state_dict()[k].cpu(),v.cpu()) for k,v in subset.items())
    assert all(n.startswith(("temporal_transformer.","policy_head.")) for n,p in p.named_parameters() if p.requires_grad)
    return DDP(p,device_ids=[0],output_device=0)

def stage_loader(c,stage):
    current=list(range(5*(stage-1),5*stage))
    paths=sorted((O/"ra_coreset/buffer").glob("*.pt"))
    for t in current:paths+=sorted((O/"features"/str(t)).glob("*.pt"))
    random.Random(SEED).shuffle(paths)
    dataset=FileTrajectoryDataset([str(p) for p in paths])
    sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True,seed=0)
    loader=DataLoader(dataset,batch_size=10,sampler=sampler,num_workers=c.train.num_workers,
                      persistent_workers=False)
    return current,dataset,sampler,loader

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    verify();assert (O/"FEATURES.json").exists()
    assert json.loads((O/"GPU_PREFLIGHT.json").read_text())["status"]=="PASS"
    root=O/"ra_coreset";root.mkdir(exist_ok=True);cp=root/"recovery/latest.pth"
    assert not (root/"TRAINING_COMPLETE.json").exists()
    assert args.resume or not cp.exists()
    old=torch.load(cp,map_location="cpu") if args.resume else None
    dist.init_process_group("nccl");torch.cuda.set_device(0)
    try:
        c=cfg("train");model=build(c);steps=0;history=[]
        if old:
            assert old["contract_sha256"]==sha(O/"CONTRACT.json")
            model.load_state_dict(old["state"]["model"],strict=True)
            restore_rng(old["state"]["rng"])
            steps=old["steps"];history=old["history"]
            actual=buffer_hashes()
            assert all(actual.get(n)==h for n,h in old["buffer_hashes"].items())
            if actual!=old["buffer_hashes"]:assert old["next_epoch"]==100 and not old["stage_complete"]
        else:assert not buffer_hashes()
        for stage in range(1,6):
            if old and (stage<old["stage"] or stage==old["stage"] and old["stage_complete"]):continue
            start=old["next_epoch"] if old and stage==old["stage"] else 0
            current=list(range(5*(stage-1),5*stage))
            if start==100:
                for task in current:store(task)
                old["stage_complete"]=True;old["buffer_hashes"]=buffer_hashes();save(cp,old)
                continue
            current,dataset,sampler,loader=stage_loader(c,stage)
            info=grouped_ifa.select(current,O/"features",root/"buffer")
            refs,deviation=grouped_ifa.references(current,O/"features",root/"buffer")
            write(root/("ifa_stage%d.json"%stage),dict(info=info,reference_max_deviation=deviation,
                accessible_paths=dataset.file_paths))
            opt=torch.optim.AdamW(model.parameters(),**c.train.optimizer.kwargs)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,make_linear_schedule_with_warmup(500,100*len(loader)))
            if old and stage==old["stage"]:restore(model,opt,sched,old["state"])
            for epoch in range(start,100):
                sampler.set_epoch(epoch);model.train();begin=time.monotonic();total=0.
                for payload,tasks in loader:
                    opt.zero_grad();x=payload["feature"].squeeze(1).cuda()
                    target=payload["actions"].squeeze(1).cuda()
                    distribution,last=model(x)
                    bc=model.module.policy_head.loss_fn(distribution,target)
                    ids=torch.tensor([int(t) for t in tasks],device=last.device)
                    extra=grouped_ifa.loss(last,ids,info,refs,.1)
                    value=bc+.1*extra;assert torch.isfinite(value)
                    (c.train.loss_scale*value).backward()
                    if c.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip)
                    opt.step();sched.step();steps+=1;total+=float(value.detach())
                history.append(dict(stage=stage,epoch=epoch,batches=len(loader),steps=steps,
                                    loss=total/len(loader),seconds=time.monotonic()-begin))
                if epoch==99:
                    save(root/"models"/("mlrifa_stage%d_ep99.pth"%stage),dict(
                        state_dict=cpu(model.module.state_dict()),cfg=c,stage=stage,epoch=99,
                        contract_sha256=sha(O/"CONTRACT.json"),pretrain_sha256=sha(c.pretrain)))
                save(cp,dict(state=capture(model,opt,sched),stage=stage,next_epoch=epoch+1,
                    stage_complete=False,steps=steps,history=history,buffer_hashes=buffer_hashes(),
                    contract_sha256=sha(O/"CONTRACT.json")))
                write(root/"epochs.json",history)
                write(O/"RUN_STATUS.json",dict(status="TRAINING",stage=stage,epoch=epoch,steps=steps,pid=os.getpid(),unix=time.time()))
                print(json.dumps(history[-1]),flush=True)
            for task in current:store(task)
            save(cp,dict(state=capture(model,opt,sched),stage=stage,next_epoch=100,
                stage_complete=True,steps=steps,history=history,buffer_hashes=buffer_hashes(),
                contract_sha256=sha(O/"CONTRACT.json")))
            old=None
        assert len(history)==500 and steps==sum(r["batches"] for r in history)
        write(root/"epochs.json",history)
        write(root/"TRAINING_COMPLETE.json",dict(epochs=500,steps=steps,
            checkpoints={p.name:sha(p) for p in sorted((root/"models").glob("*.pth"))},
            summed_epoch_seconds=sum(r["seconds"] for r in history)))
    finally:dist.destroy_process_group()
if __name__=="__main__":main()
