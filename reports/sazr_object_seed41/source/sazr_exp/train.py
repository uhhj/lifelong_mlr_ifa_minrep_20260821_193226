import argparse
from .common import *
from torch.utils.data import DataLoader,DistributedSampler
from libero.lifelong.algos.mlr import build_dataset_for_task

@torch.no_grad()
def structure(p,bank,anchors,dataset,label,ph):
    assert tensor_hash(p)==ph
    e=torch.from_numpy(np.load(W/"language.npz")["canonical"][6:7]).cuda().repeat(len(anchors),1)
    response=bank(anchors.cuda(),e)
    residual=float(response.abs().max());assert residual<=1e-6
    z=anchors.cuda().double();q=bank.Q
    relative=float((z-z@q@q.T).norm()/z.norm());assert relative<=1e-10
    maximum=0.;off=0.
    for samples,tasks in DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0):
        h=samples["feature"].squeeze(1).cuda();out=decode(p,bank,h)
        maximum=max(maximum,float(out["delta"].abs().max()))
        mask=torch.ones_like(out["l"],dtype=torch.bool);mask.scatter_(1,out["selected"][:,None],False)
        assert torch.equal(out["means"][mask],out["b"].tanh()[mask])
        zero=corrected(out["b"],out["s"],out["l"],torch.zeros_like(out["delta"]))
        assert torch.equal(zero["means"],out["b"].tanh())
    # Anchor output is zero regardless of trained UV; check actual mean effect too.
    b,s,l=raw(p.policy_head,anchors.cuda())
    means=corrected(b,s,l,response)["means"]
    mean_error=float((means-b.tanh()).abs().max());assert mean_error<=1e-6
    row=dict(label=label,anchor_response_max=residual,anchor_mean_error=mean_error,projection_relative_error=relative,
        dataset_delta_max=maximum,U_norm=float(bank.U.norm()),V_norm=float(bank.V.norm()),public_hash=ph,
        samples=len(dataset),nonselected_exact=True,zero_parent_exact=True,passed=True)
    append(O/"STRUCTURE.jsonl",row);return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    verify();setup()
    for f,h in js(O/"ANCHOR_CONTRACT.json")["files"].items():assert sha(f)==h
    assert not (O/"CALIBRATED.json").exists()
    recovery=O/"recovery/latest.pth";assert args.resume or not recovery.exists()
    p=new_public();ph=tensor_hash(p);aq=torch.load(O/"anchor_basis.pth",map_location="cpu")
    bank=make_bank(aq["Q"]);opt,sch=optimizer(bank)
    empty=O/"empty_buffer";empty.mkdir(exist_ok=True);assert not list(empty.iterdir())
    ds=build_dataset_for_task(str(P/"features"),6,str(empty),41,start=6);assert len(ds)==1562
    sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0)
    loader=DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0,drop_last=False)
    assert len(loader)==157
    steps=0;epochs=[];start=0
    if args.resume:
        cp=torch.load(recovery,map_location="cpu");assert cp["contract_sha256"]==sha(O/"CONTRACT.json")
        assert cp["anchor_contract_sha256"]==sha(O/"ANCHOR_CONTRACT.json")
        bank.load_state_dict(cp["bank"]);opt.load_state_dict(cp["optimizer"]);sch.load_state_dict(cp["scheduler"])
        for n,v in bank.named_parameters():v.grad=None if cp["grads"][n] is None else cp["grads"][n].cuda()
        for n,m in bank.named_modules():m.training=cp["modes"][n]
        restore_rng(cp["rng"]);steps=cp["steps"];epochs=cp["epochs"];start=cp["next_epoch"]
    else:
        assert not (O/"epochs.json").exists()
        structure(p,bank,aq["z"],ds,"start",ph)
        control_seed(41)
    for epoch in range(start,10):
        tic=time.time();sampler.set_epoch(epoch);losses=[]
        for bi,(samples,tasks) in enumerate(loader):
            h=samples["feature"].squeeze(1).cuda();a=samples["actions"].squeeze(1).cuda()
            r=update(p,bank,opt,sch,h,a);steps+=1;losses.append(r["loss"])
            append(O/"batches.jsonl",dict(epoch=epoch,batch=bi,logical_step=steps,**r))
        torch.cuda.synchronize()
        structure(p,bank,aq["z"],ds,f"epoch{epoch}",ph)
        epochs.append(dict(epoch=epoch,batches=157,samples=1562,steps=steps,loss=float(np.mean(losses)),seconds=time.time()-tic))
        write(O/"epochs.json",epochs)
        cp=dict(bank=cpu(bank.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(sch.state_dict()),
            grads={n:cpu(v.grad) for n,v in bank.named_parameters()},modes={n:m.training for n,m in bank.named_modules()},
            rng=cpu(rng()),next_epoch=epoch+1,steps=steps,epochs=epochs,public_hash=ph,
            contract_sha256=sha(O/"CONTRACT.json"),anchor_contract_sha256=sha(O/"ANCHOR_CONTRACT.json"))
        save(recovery,cp);status("CALIBRATING",epoch=epoch,steps=steps)
        if epoch==9:save(O/"models/sazr_stage6_calibration_ep9.pth",cp)
    assert steps==1570 and len(epochs)==10
    verify();write(O/"CALIBRATED.json",dict(status="PASS",epochs=10,steps=1570,exposures=15620,
        checkpoint_sha256=sha(O/"models/sazr_stage6_calibration_ep9.pth"),public_hash=ph))
    print("CALIBRATED 10epoch/1570steps")
if __name__=="__main__":main()
