from .model import *

@torch.no_grad()
def check(p,t0,bank,dataset,label,public_hash,t0_hash):
    state=cpu(rng());start=time.time();maxgate=0.;selferr=0.;cross=0.;amp=0.;sat=0;count=0
    from torch.utils.data import DataLoader
    try:
        assert tensor_hash(p)==public_hash and tensor_hash(t0)==t0_hash
        for samples,tasks in DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0):
            h=samples['feature'].squeeze(1).cuda();ids=torch.tensor([int(t) for t in tasks],device='cuda')
            out=decode(p,t0,bank,h);e=h[:,-1,0].double();response=(e/e.norm(dim=-1,keepdim=True))@bank.gate_w.T
            expected=torch.stack([ids==t for t in bank.task_ids],1)
            if (~expected).any():maxgate=max(maxgate,float(response[~expected].abs().max()))
            selferr=max(selferr,float((response[expected]-1).abs().max()))
            values=torch.stack([bank.branches[str(t)](out['q']).double()*response[:,j,None] for j,t in enumerate(bank.task_ids)],1)
            total=values.sum(1)
            for j,t in enumerate(bank.task_ids):
                foreign=ids!=t
                if foreign.any():
                    without=(.1*torch.tanh(total-values[:,j])).float()
                    diff=(out['delta'][foreign]-without[foreign]).abs().max();cross=max(cross,float(diff))
                    means=(out['b'][foreign]+without[foreign,None,:]).tanh()
                    cross=max(cross,float((out['means'][foreign]-means).abs().max()))
            amp=max(amp,float(out['delta'].abs().max()));sat+=int((out['delta'].abs()>=.099).sum());count+=len(h)*7
            assert float((out['means']-out['base_means']).abs().max())<=.100001
        report=dict(label=label,stage=max(bank.task_ids),samples=len(dataset),off_gate=maxgate,self_gate_error=selferr,cross_contribution_and_mean=cross,delta_max=amp,saturation_fraction=sat/count,public_hash=public_hash,T0_hash=t0_hash,gate_hash=hashlib.sha256(bank.gate_w.cpu().numpy().tobytes()).hexdigest(),branch_norms={n:{k:float(v.norm()) for k,v in branch.named_parameters()} for n,branch in bank.branches.items()},seconds=time.time()-start)
        report['passed']=maxgate<=1e-6 and selferr<=1e-5 and cross<=1e-6
        append(O/'STRUCTURE.jsonl',report);assert report['passed'],('FAIL_RUNTIME',report)
        return report
    finally:restore_rng(state)
