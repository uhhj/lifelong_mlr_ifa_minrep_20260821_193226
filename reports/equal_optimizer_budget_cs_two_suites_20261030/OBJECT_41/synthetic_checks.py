"""Compare against archived implementation, not a restatement in the upload."""
import torch
from losses import aux, reference_r
from legacy_losses import aux as archived_c
from directional_prototype import auxiliary_v3, segment_distance_squared


def run():
    torch.manual_seed(20261025)
    checked = 0
    for dtype in [torch.float32, torch.float64]:
        for tie in [False, True]:
            q = (torch.randn(13,5,7,dtype=dtype,requires_grad=True),
                 torch.full((13,5,7),.2,dtype=dtype,requires_grad=True),
                 torch.randn(13,5,dtype=dtype,requires_grad=True))
            if tie:
                with torch.no_grad(): q[0][:] = q[0][:,:1].clone()
            t = tuple(x.detach().clone().requires_grad_() for x in q)
            y = torch.randn(13,7,dtype=dtype)
            scale = torch.tensor([.4,.1,1.],dtype=dtype)
            w = torch.linspace(.1,.9,13,dtype=dtype,requires_grad=True)
            old = archived_c('C',q,y,t,scale)[0]
            for val in [aux('C',q,y,t,scale,w)[0], aux('G',q,y,t,scale,w,0.)[0]]:
                assert torch.equal(val,old)
                gg=torch.autograd.grad(val,q,retain_graph=True)
                go=torch.autograd.grad(old,q,retain_graph=True)
                assert all(torch.equal(a,b) for a,b in zip(gg,go))
                checked += 1
            r = aux('R',q,y,t,scale,w)[0]
            rr = reference_r(q,y,t,scale,w)
            assert torch.equal(r,rr)
            assert all(torch.equal(a,b) for a,b in zip(torch.autograd.grad(r,q,retain_graph=True),torch.autograd.grad(rr,q,retain_graph=True)))
            for arm in 'RG':
                loss = aux(arm,q,y,t,scale,w)[0]
                grads = torch.autograd.grad(loss,(*q,*t,w),allow_unused=True,retain_graph=True)
                assert all(g is not None and torch.isfinite(g).all() for g in grads[:3])
                assert all(g is None for g in grads[3:])
    p=torch.randn(4096,14,dtype=torch.float64,requires_grad=True)
    t=torch.randn_like(p);e=torch.randn_like(p)
    r,_=segment_distance_squared(p,t,e);c=(p-e).square().sum(-1)
    gr=torch.autograd.grad(r.sum(),p,retain_graph=True)[0]
    gc=torch.autograd.grad(c.sum(),p)[0]
    assert (r-c).max()<1e-10 and ((gr*gc).sum(-1)-4*r).min()>-1e-10
    return dict(status='PASS_SYNTHETIC_NOT_REAL_ADMISSION',archived_C_exact_loss_and_output_gradient_cases=checked,
                R_independent_formula_and_gradient_exact=True,teacher_weight_no_gradient=True,
                projection_bound_and_gradient_vectors=4096,real_checkpoints=0,optimizer_calls=0,environments=0)

if __name__=='__main__':
    import json
    from pathlib import Path
    result=run();result['torch_version']=torch.__version__
    Path(__file__).with_name('INDEPENDENT_CPU_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
