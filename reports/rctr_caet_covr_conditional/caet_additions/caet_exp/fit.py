"""Ten fixed auxiliary epochs on current-task plus actually retained RA labels."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from coreset_exp.common import O, SEED, verify, sha, save, write, rng, restore_rng
from coreset_exp.model import cfg
from libero.lifelong.algos.mlr import build_dataset_for_task
from libero.lifelong.utils import control_seed
from .dataset import EffectDataset
from .model import EffectModel
from .transfer import pair_masks, transfer_loss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("task",type=int,choices=(7,8,9))
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    verify()
    root=O/"effects/models"/("task%d" % args.task)
    assert not (root/"ACCEPTANCE.json").exists()
    assert args.resume or not root.exists(), "Explicit auxiliary epoch resume only"
    root.mkdir(parents=True,exist_ok=True)
    c=cfg("ra_coreset")
    ds=EffectDataset(build_dataset_for_task(c.feature_dir,args.task,c.buffer_dir,SEED,start=6),args.task)
    payload=[ds[i] for i in range(len(ds)) if ds.labels[str(ds.file_paths[i])]["valid"]]
    features=torch.stack([r[0]["feature"][0] for r in payload])
    actions=torch.stack([r[0]["actions"][0,-1] for r in payload])
    effects=torch.stack([r[0]["effect_target"] for r in payload])
    moving=torch.tensor([r[0]["effect_moving"] for r in payload],dtype=torch.bool)
    tasks=torch.tensor([int(r[1]) for r in payload])
    holdout=torch.tensor([r[0]["effect_holdout"] for r in payload],dtype=torch.bool)
    train_ids=(~holdout).nonzero().flatten()
    val_ids=holdout.nonzero().flatten()
    assert len(train_ids)>0 and len(val_ids)>0
    # Training-only population scale per physical coordinate. Floors prevent division
    # by near-zero dimensions; no fitted statistics from held-out trajectories.
    center=effects[train_ids].double().mean(0).float()
    scale=effects[train_ids].double().std(0,unbiased=False).float().clamp_min(1e-6)
    normalized=(effects-center)/scale
    positive,negative=pair_masks(normalized[train_ids],moving[train_ids],tasks[train_ids])
    assert int(positive.sum())>0 and int(negative.sum())>0
    control_seed(810000000+1000*SEED+args.task)
    model=EffectModel().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    cp=root/"recovery.pth"
    start=0; history=[];steps=0
    if args.resume:
        r=torch.load(cp,map_location="cpu")
        assert r["contract_sha256"]==sha(O/"CONTRACT.json")
        assert torch.equal(r["center"],center) and torch.equal(r["scale"],scale)
        model.load_state_dict(r["model"],strict=True)
        optimizer.load_state_dict(r["optimizer"])
        restore_rng(r["rng"])
        start,steps,history=r["next_epoch"],r["steps"],r["history"]
    def batch(ids):
        return [x[ids].cuda() for x in (features,actions,normalized,moving,tasks)]
    for epoch in range(start,10):
        model.train()
        g=torch.Generator().manual_seed(820000000+10000*SEED+100*args.task+epoch)
        order=train_ids[torch.randperm(len(train_ids),generator=g)]
        total=0.;used=0;began=time.monotonic()
        for k in range(0,len(order),64):
            x,a,e,m,t=batch(order[k:k+64])
            optimizer.zero_grad()
            prediction,embedding=model(x,a)
            physical=F.smooth_l1_loss(prediction,e)
            contrast,pairs=transfer_loss(embedding,e,m,t)
            loss=physical+.1*contrast
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            steps+=1;used+=pairs;total+=float(loss.detach())
        model.eval()
        with torch.no_grad():
            errors=[]
            for k in range(0,len(val_ids),64):
                x,a,e,_,_=batch(val_ids[k:k+64])
                pred,_=model(x,a)
                errors.append((pred-e).abs().cpu())
            mae=torch.cat(errors).mean(0).tolist()
        history.append(dict(epoch=epoch,steps=steps,
            loss=total/math.ceil(len(train_ids)/64),actual_cross_task_positive_pairs=used,
            heldout_normalized_mae=mae,seconds=time.monotonic()-began))
        save(cp,dict(next_epoch=epoch+1,steps=steps,history=history,
            model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),
            center=center,scale=scale,contract_sha256=sha(O/"CONTRACT.json")))
        write(root/"epochs.json",history)
    assert steps==10*math.ceil(len(train_ids)/64)
    assert sum(r["actual_cross_task_positive_pairs"] for r in history)>0
    # The fitted model is frozen for the policy, but dF/da must remain available.
    model.eval().requires_grad_(False)
    ids=val_ids[:64]
    x,a,e,_,_=batch(ids)
    a=a.detach().requires_grad_(True)
    prediction,_=model(x,a)
    F.smooth_l1_loss(prediction,e).backward()
    assert torch.isfinite(a.grad).all() and float(a.grad.abs().sum())>0
    save(root/"final.pth",dict(model=model.state_dict(),center=center,scale=scale,
        input_paths=[str(p) for p in ds.file_paths],contract_sha256=sha(O/"CONTRACT.json")))
    write(root/"ACCEPTANCE.json",dict(status="PASS",task=args.task,epochs=10,optimizer_steps=steps,
        train_windows=len(train_ids),heldout_windows=len(val_ids),
        global_positive_pairs=int(positive.sum()),global_negative_pairs=int(negative.sum()),
        observed_training_positive_pairs=sum(r["actual_cross_task_positive_pairs"] for r in history),
        heldout_final_normalized_mae=history[-1]["heldout_normalized_mae"],
        action_gradient_l1=float(a.grad.abs().sum()),action_gradient_finite=True,
        frozen_predictor_sha256=sha(root/"final.pth"),source_dataset_windows=len(ds),
        policy_updates=0,training_only_scale_floor=1e-6,
        label_distribution="current full task plus actual RA memory; whole-demo holdout",
        validation_is_not_closed_loop_evidence=True))


if __name__=="__main__":
    main()
