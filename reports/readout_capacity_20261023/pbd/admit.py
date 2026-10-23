import copy,argparse
from .common import *
from .models import build
from torch.nn import functional as F

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['capacity']);a=ap.parse_args();verify();d,x=data(6);ii=torch.where(d['mask'][:,0])[0][:4];x=x[ii].cuda();target=d['q'][ii].cuda()/SCALE.cuda();rows=[]
 kinds=['P0','G','S']
 # Untrained parameterization proof: same seed, complete state, RNG consumption, output.
 import importlib.util
 spec=importlib.util.spec_from_file_location('pbd.reference_v2_models',str(V2/'pbd/models.py'));reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
 equivalence=[]
 for k in kinds:
  seed(647);old=reference_module.build(k).cuda();old_rng=torch.get_rng_state().clone();old_cuda=[r.clone() for r in torch.cuda.get_rng_state_all()]
  seed(647);new512=build(k,temporal_ffn=512).cuda()
  assert torch.equal(old_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(old_cuda,torch.cuda.get_rng_state_all()))
  assert old.state_dict().keys()==new512.state_dict().keys() and all(torch.equal(v,new512.state_dict()[n]) for n,v in old.state_dict().items())
  pa=x[:,:,1:3,None,:].expand(-1,-1,-1,196,-1).clone() if k=='S' else None
  with torch.no_grad():
   ref_output=old(x,pa) if k in ('G','S') else old(x);new_output=new512(x,pa) if k in ('G','S') else new512(x)
  assert torch.equal(ref_output,new_output)
  equivalence.append(dict(kind=k,state_exact=True,rng_exact=True,output_exact=True,parameters=sum(p.numel() for p in old.parameters()),untrained_forwards=2))
  del old,new512,pa

 for kind in kinds:
  seed(647);m=build(kind).cuda();assert sum(p.numel() for p in m.parameters())==({'P0':483335,'G':400775,'S':400775}[kind]);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
  patch=x[:,:,1:3,None,:].expand(-1,-1,-1,196,-1).clone() if kind=='S' else None
  def call(net,xx,pp=patch):return net(xx,pp) if kind in ('G','S') else net(xx)
  if kind=='S':
   seed(647);base=build('G').cuda();assert torch.equal(call(m,x),base(x));del base
  inp=x.detach().clone().requires_grad_(True);pa=patch.detach().clone().requires_grad_(True) if patch is not None else None
  y=m(inp,pa) if kind in ('G','S') else m(inp);y.square().sum().backward()
  visual=float(pa.grad[:,-1].abs().sum()) if pa is not None else float(inp.grad[:,-1,1:3].abs().sum())
  assert (visual==0) if kind=='P0' else (visual>0)
  for n in range(1,9):assert torch.isfinite(m(x[:,:n],patch[:,:n] if patch is not None else None) if kind in ('G','S') else m(x[:,:n])).all()
  def step(net,op):
   op.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(call(net,x),target);assert torch.isfinite(loss);loss.backward();op.step();return float(loss)
  step(m,opt)
  cp=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),grads={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()},modes={n:v.training for n,v in m.named_modules()},requires_grad={n:p.requires_grad for n,p in m.named_parameters()})
  reference=step(m,opt);new=build(kind).cuda();new.load_state_dict(cp['model'],strict=True);op=torch.optim.AdamW(new.parameters(),lr=3e-4,weight_decay=.01);op.load_state_dict(cp['optimizer'])
  for n,p in new.named_parameters():p.requires_grad_(cp['requires_grad'][n]);p.grad=cp['grads'][n]
  for n,v in new.named_modules():v.training=cp['modes'][n]
  torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng']);restored=step(new,op)
  assert reference==restored and all(torch.equal(v,new.state_dict()[n]) for n,v in m.state_dict().items())
  rows.append(dict(kind=kind,parameters=sum(p.numel() for p in m.parameters()),latest_visual_gradient_l1=visual,isolated_optimizer_calls=3,restore_exact=True,finite_prefix_lengths=list(range(1,9))))
 verify();write(OUT/(a.phase.upper()+'_ADMISSION.json'),dict(status='PASS_ISOLATED_GPU',rows=rows,baseline_512_equivalence=equivalence,baseline_untrained_forwards=6,temporal_ffn=128,optimizer_calls=3*len(kinds),environment_episodes=0));log('admission',phase=a.phase,optimizer_calls=3*len(kinds))
if __name__=='__main__':main()
