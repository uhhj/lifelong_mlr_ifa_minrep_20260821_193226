import torch
class ModelFisher:
 def __init__(self,engine,draws):
  self.e=engine;self.draws=draws.detach();B,S,D=draws.shape
  self.z=engine.z['old_build'].repeat_interleave(S,0)
  self.a=draws.reshape(B*S,D);self.calls=0
 def scores_output(self,t):return -self.e.call(t,self.z,self.a)
 def mv(self,v):
  self.calls+=1;v=torch.as_tensor(v,dtype=self.e.theta.dtype,device=self.e.theta.device)
  j=torch.autograd.functional.jvp(self.scores_output,self.e.theta,v,create_graph=False)[1]
  return torch.autograd.functional.vjp(self.scores_output,self.e.theta,j/len(j),create_graph=False)[1].detach()

def sample(engine,seed=2026092306,n=8):
 # A private generator leaves global Python/NumPy/torch/CUDA streams alone.
 gen=torch.Generator(device=engine.theta.device);gen.manual_seed(seed)
 with torch.no_grad():
  means,stds,logits=engine.module.head.forward_fn(engine.z['old_build'])
  modes=torch.multinomial(logits.softmax(-1),n,replacement=True,generator=gen)
  ix=modes[:,:,None].expand(-1,-1,means.shape[-1]);mu=means.gather(1,ix);sigma=stds.gather(1,ix)
  noise=torch.randn(mu.shape,device=mu.device,dtype=mu.dtype,generator=gen)
  draws=(mu+sigma*noise).detach()
 return draws,modes
