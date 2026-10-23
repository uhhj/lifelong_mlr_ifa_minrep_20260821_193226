import argparse
from .common import *
from .readout_models import LightReadout

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',default=41,type=int);ap.parse_args();verify();seed(41)
 m=temporal();x=torch.load(OUT/'private/native6.pt')[0:1].cuda();rows=[]
 def encode(v):return m(v.reshape(len(v),32,512))
 base=encode(x);direction=.2*torch.sin(torch.arange(512,device='cuda')*.13)
 for indices in ([29,30],[31],[25,26],[27]):
  y=x.clone().reshape(1,32,512);y[:,indices]+=direction;z=encode(y)
  change=[float((z[:,i]-base[:,i]).abs().max()) for i in (28,31)]
  if min(indices)>28:assert change[0]==0
  else:assert change[0]>0
  assert change[1]>0;rows.append(dict(indices=indices,old_change=change[0],full_change=change[1]))
 v=x.clone().requires_grad_(True);z=encode(v);proj=torch.sin(torch.arange(512,device='cuda')*.17)
 grads={}
 for idx in (28,31):
  g=torch.autograd.grad((z[:,idx]*proj).sum(),v,retain_graph=True)[0].abs().sum(-1)[0];grads[str(idx)]=g.cpu().tolist()
 assert all(grads['28'][-1][k]==0 for k in (1,2,3)) and all(grads['31'][-1][k]>0 for k in (1,2,3))
 light=LightReadout().cuda().eval();v=x.clone().requires_grad_(True);g=torch.autograd.grad((light.context(v)*torch.sin(torch.arange(128,device='cuda'))).sum(),v)[0].abs().sum(-1)[0]
 assert all(float(g[-1,k])>0 for k in (1,2,3))
 # Native history starts at one frame, no future padding. Exact shape check at each legal length.
 lengths=[]
 for n in range(1,9):
  assert m(x[:,-n:].reshape(1,4*n,512)).shape==(1,4*n,512);assert light(x[:,-n:]).shape==(1,7);lengths.append(n)
 write(OUT/'VISIBILITY_AUDIT.json',dict(status='PASS',official_sha256=sha(config().pretrain),strict_temporal_keys=len(m.state_dict()),changes=rows,input_gradient_norms=grads,p3_gradient_norms=g.cpu().tolist(),history_lengths=lengths,optimizer_calls=0,environment_steps=0))
 log('visibility_complete')
if __name__=='__main__':main()
