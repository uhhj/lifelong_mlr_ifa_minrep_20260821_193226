import torch

def apply(parameters,before,method,d64,d32,correction32,native_parameters):
 """W1/W2 start at before: no intermediate native parameter load in either branch."""
 parameters=tuple(parameters);before=tuple(before)
 with torch.no_grad():
  if method=='W0':
   offset=0
   for p,native in zip(parameters,native_parameters):
    p.copy_(native);c=correction32[offset:offset+p.numel()].reshape_as(p);p.copy_(p.detach()+(-c));offset+=p.numel()
  else:
   assert method in ('W1','W2')
   # Deliberately build the full double target, measuring the requested overhead.
   theta=torch.cat([v.reshape(-1) for v in before]).double()
   target=theta+(d64 if method=='W1' else d32.double())
   offset=0
   for p in parameters:p.copy_(target[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
  return torch.cat([(p.detach().double()-b.double()).reshape(-1) for p,b in zip(parameters,before)])

def bins(ratio):
 edges=torch.tensor([.5,1.,2.,4.],dtype=torch.float64,device=ratio.device)
 index=torch.bucketize(ratio,edges,right=True);return [int((index==k).sum()) for k in range(5)]
