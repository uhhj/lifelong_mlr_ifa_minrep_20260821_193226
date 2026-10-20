import torch
from torch import nn
from torch.distributions import Independent, Normal

def basis(z):
    assert z.ndim==2 and z.shape[1]==512 and 0<len(z)<=32
    _,s,vh=torch.linalg.svd(z.double(),full_matrices=False)
    threshold=max(1e-12,1e-10*float(s.max()))
    q=vh[s>threshold].T.contiguous()
    assert 0<q.shape[1]<=32
    err=(z.double()-z.double()@q@q.T).norm()/z.double().norm()
    assert float(err)<=1e-10
    return q,dict(rank=q.shape[1],singular_values=s.tolist(),threshold=threshold,relative_error=float(err))

class Residual(nn.Module):
    def __init__(self,q,canonical,gate):
        super().__init__()
        self.register_buffer("Q",q.detach().double())
        canonical=canonical.double()
        self.register_buffer("canonical",canonical/canonical.norm(dim=-1,keepdim=True))
        self.register_buffer("gate",gate.double())
        gen=torch.Generator(device="cpu").manual_seed(47)
        self.V=nn.Parameter(torch.randn(8,512,generator=gen)*.02)
        self.U=nn.Parameter(torch.zeros(7,8))
    def route(self,e):
        ee=e.double();norm=ee.norm(dim=-1,keepdim=True)
        ok=norm[:,0]>1e-12
        normalized=ee/norm.clamp_min(1e-12)
        matches=(normalized[:,None,:]-self.canonical[None,:,:]).abs().amax(-1)<=1e-6
        return ok & (matches.sum(-1)==1) & matches[:,6]
    def forward(self,z,e):
        zz=z.double();projected=zz-zz@self.Q@self.Q.T
        raw=(projected.to(self.V.dtype)@self.V.T)@self.U.T
        response=(e.double()/e.double().norm(dim=-1,keepdim=True).clamp_min(1e-12))@self.gate
        return (.1*torch.tanh(raw.double()*response[:,None])).to(z.dtype)

def corrected(b,s,l,delta):
    idx=l.argmax(-1);rr=torch.arange(len(b),device=b.device)
    means=b.tanh();means=means.clone()
    means[rr,idx]=(b[rr,idx]+delta).tanh()
    return dict(means=means,selected=idx,dist=Independent(Normal(means[rr,idx],s[rr,idx]),1))

def noise_table(indices):
    # No process-global RNG consumption and no dependence on outcome/termination.
    return torch.stack([torch.randn(600,7,generator=torch.Generator(device="cpu").manual_seed(41000000+i)) for i in indices])

def choose_anchor_times(lengths,total=32):
    counts=[0]*len(lengths)
    for _ in range(min(total,sum(lengths))):
        eligible=[i for i,n in enumerate(lengths) if counts[i]<n]
        i=min(eligible,key=lambda j:(counts[j],j));counts[i]+=1
    result=[]
    for n,k in zip(lengths,counts):
        ids=[0] if k==1 else [i*(n-1)//(k-1) for i in range(k)]
        assert len(set(ids))==k
        result.append(ids)
    return result
