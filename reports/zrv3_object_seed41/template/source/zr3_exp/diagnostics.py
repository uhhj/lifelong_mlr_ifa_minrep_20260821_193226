from .model import *

@torch.no_grad()
def check(p,t0,bank,dataset,label,public_hash,t0_hash):
    state=cpu(rng());start=time.time();maxgate=0.;selferr=0.;cross=0.;amp=0.;sat=0;count=0;maxkl=0.;rawkl=0.;projected=0
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
                    means=calibrate_distribution(out['b'][foreign],out['scales'][foreign],out['logits'][foreign],without[foreign],EPSILON)['means']
                    cross=max(cross,float((out['means'][foreign]-means).abs().max()))
            maxkl=max(maxkl,float(out['actual_kl'].max()));rawkl=max(rawkl,float(out['raw_kl'].max()));projected+=int((out['alpha']<1).sum())
            off=torch.ones_like(out['logits'],dtype=torch.bool);off.scatter_(1,out['selected'][:,None],False)
            assert torch.equal(out['means'][off],out['base_means'][off])
            zero=calibrate_distribution(out['b'],out['scales'],out['logits'],torch.zeros_like(out['delta']),EPSILON)
            assert torch.equal(zero['means'],out['base_means']) and bool((zero['actual_kl']==0).all())
            assert torch.equal(out['selected'],out['logits'].argmax(-1))
            amp=max(amp,float(out['delta'].abs().max()));sat+=int((out['delta'].abs()>=.099).sum());count+=len(h)*7
            assert float((out['means']-out['base_means']).abs().max())<=.100001
        report=dict(label=label,stage=max(bank.task_ids),samples=len(dataset),off_gate=maxgate,self_gate_error=selferr,cross_contribution_and_mean=cross,delta_max=amp,saturation_fraction=sat/count,public_hash=public_hash,T0_hash=t0_hash,gate_hash=hashlib.sha256(bank.gate_w.cpu().numpy().tobytes()).hexdigest(),branch_norms={n:{k:float(v.norm()) for k,v in branch.named_parameters()} for n,branch in bank.branches.items()},seconds=time.time()-start)
        report.update(actual_kl_max=maxkl,raw_kl_max=rawkl,projected_samples=projected,epsilon=EPSILON,
                      unselected_means_exact=True,zero_residual_parent_exact=True,selected_component_exact=True)
        report['passed']=maxgate<=1e-6 and selferr<=1e-5 and cross<=1e-6 and (EPSILON is None or maxkl<=EPSILON+1e-12)
        append(O/'STRUCTURE.jsonl',report);assert report['passed'],('FAIL_RUNTIME',report)
        return report
    finally:restore_rng(state)
