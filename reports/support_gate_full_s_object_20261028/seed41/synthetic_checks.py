"""Numerical checks for this candidate only; not real-model or environment acceptance."""
import torch
from losses import expert,retain
from legacy_losses import aux as legacy_aux

def run():
    torch.manual_seed(41)
    mu=torch.randn(6,5,7,requires_grad=True);sd=torch.full_like(mu,.2,requires_grad=True);z=torch.randn(6,5,requires_grad=True)
    q=(mu,sd,z);target=torch.randn(6,7);scale=torch.tensor([.4,.1,1.]);teacher=tuple(x.detach().clone().requires_grad_(True) for x in q)
    value,_,_=expert(q,target,scale);reference,_=legacy_aux('C',q,target,tuple(x.detach() for x in q),scale)
    assert torch.equal(value,reference)
    g0=torch.autograd.grad(value,q,retain_graph=True);g1=torch.autograd.grad(reference,q,retain_graph=True)
    assert all(torch.equal(a,b) for a,b in zip(g0,g1))
    ix=torch.arange(6);tk=teacher[2].argmax(-1);tm=teacher[0][ix,tk];ts=teacher[1][ix,tk]
    a,b=retain(q,tm,ts,scale);loss=(a+.1*b).mean();loss.backward()
    assert all(x.grad is None for x in teacher)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in q)
    perm=torch.tensor([3,1,4,0,2]);aa,bb=retain((mu[:,perm],sd[:,perm],z[:,perm]),tm,ts,scale)
    assert torch.allclose(a,aa,atol=1e-6,rtol=1e-6) and torch.allclose(b,bb,atol=1e-6,rtol=1e-6)
    return dict(status='PASS_SYNTHETIC_ONLY',C_loss_and_output_gradient_exact=True,teacher_detached=True,finite_student_gradients=True,component_permutation_invariance=True,model_loads=0,environment_episodes=0)

if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
