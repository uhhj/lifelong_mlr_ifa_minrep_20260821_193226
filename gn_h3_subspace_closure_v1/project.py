import torch
def project(U,v):return U@torch.linalg.solve(U.T@U,U.T@v)
def orthonormal_random(n,r,seed,device,dtype=torch.float64):
 gen=torch.Generator(device=device);gen.manual_seed(seed)
 return torch.linalg.qr(torch.randn(n,r,dtype=dtype,device=device,generator=gen),mode='reduced')[0]
