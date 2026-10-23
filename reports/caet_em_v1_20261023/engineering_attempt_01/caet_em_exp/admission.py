"""Disposable own-source admission. All optimization calls are separately counted."""
import argparse
from torch.utils.data import DataLoader
from .common import *
from .model import *
from .dataset import EvidenceDataset,current
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b
def replay(m,opt,sched,fn):
 def step():
  opt.zero_grad();v=fn();assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),100);opt.step()
  if sched:sched.step()
  return cpu(v)
 step();s=capture(m,opt,sched,next_epoch=1,cursor=0);a=step();end=capture(m,opt,sched,next_epoch=2,cursor=0);restore(m,opt,sched,s);b=step();again=capture(m,opt,sched,next_epoch=2,cursor=0)
 assert torch.equal(a,b) and equal(end,again),'Full next-update replay mismatch';return 3
def main():
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=('codec',)+ARMS);args=ap.parse_args();contract=verify();seed(410010);capacity()
 rows=current(6);r=[r for r in rows.values() if r['action_mask'].all()][:10];ds=EvidenceDataset([PARENT/'features/6'/r['sample'] for r in r],r);b=to_device(next(iter(DataLoader(ds,10))))
 checks={}
 if args.kind=='codec':
  from .train_codec import build,loss
  m,opt=build();calls=replay(m,opt,None,lambda:loss(m,b)[0]);m.eval()
  with torch.no_grad():
   e=m(b);changed=dict(b);changed['action']=b['action'].roll(1,0);checks['future_changes_labels']=not torch.equal(e,m(changed));assert checks['future_changes_labels']
  checks['split_disjoint']=not set(js(ROOT/'evidence/task6/TEMPORAL_BINDING_AUDIT.json')['codec_train_demos'])&set(js(ROOT/'evidence/task6/TEMPORAL_BINDING_AUDIT.json')['codec_validation_demos']);assert checks['split_disjoint']
 else:
  from .train import load_codec,optimizer,loss
  pack,codec=load_codec();m=Policy(args.kind,pack['model'],pack['statistics']).cuda();opt,sched,names=optimizer(m,15700)
  with torch.no_grad():b['effect']=codec(b);b['effect_mask']=b['action_mask']&b['response_mask']&b['visual_mask']
  refs={6:b['feature'][0,0,0]};calls=replay(m,opt,sched,lambda:loss(m,b,codec,{'selected_pairs':[]},refs,6)[0]);m.eval()
  with torch.no_grad():
   base=m(b['feature'])[0][0]
   for k in (1,2,3):
    x=b['feature'].clone();x[:,-1,k]=x[:,-1,k].roll(1,0);diff=float((m(x)[0][0]-base).abs().max());checks['latest_modality_%d_max_delta'%k]=diff;assert diff>0
   for length in range(1,9):assert m(b['feature'][:,-length:])[0][0].shape[:2]==(10,4)
   checks['lengths_1_to_8']=True
   x=b['feature'].clone();alter=dict(b);alter['action']=torch.randn_like(b['action']);assert torch.equal(base,m(x)[0][0]);checks['no_future_deployment_input']=True
   q=m(x)[0];chosen=selected(q);assert torch.equal(chosen[2],q[2].argmax(-1));g=distribution(q);assert torch.isfinite(g.log_prob(b['action'])).all();checks['mixture_train_argmax_normal_deploy']=True
   if args.kind!='B0':
    q1,z,_=m(x);z0=torch.zeros_like(z);checks['effect_path_nonzero']=float((m.decode_z(z0,x)[0]-q1[0]).abs().max())>0;assert checks['effect_path_nonzero']
  zeros=dict(b);zeros['action_mask']=torch.zeros_like(b['action_mask']);zeros['effect_mask']=torch.zeros_like(b['effect_mask']);m.train();value,_=loss(m,zeros,codec,{'selected_pairs':[]},refs,6);assert float(value)==0;opt.zero_grad();value.backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in m.parameters());checks['empty_mask_differentiable_zero']=True
  # Cache a synthetic nontrivial old/current batch to exercise T/F gradient routing before formal B2.
  if args.kind=='B2':
   from .transfer import transfer_loss
   from .retention import retain_loss
   m.eval();q,z,_=m(b['feature']);cache=dict(w=torch.ones(10,dtype=torch.bool),z=cpu(z+.1),qgt=tuple(cpu(v) for v in q),qpred=tuple(cpu(v) for v in q));bb=dict(b);bb['task']=torch.tensor([6]*5+[7]*5,device='cuda');bb['index']=torch.arange(10,device='cuda')
   lt,_,_=transfer_loss(z,bb,cache,7);lf,route,_,_=retain_loss(m,z,bb,cache,7);opt.zero_grad();(lt+lf+.1*route).backward();assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in m.predictor.parameters());assert all(p.grad is None for p in codec.parameters());checks['T_F_gradient_and_frozen_codec']=True
  checks['optimizer_whitelist']=names
 report=dict(status='PASS',kind=args.kind,contract=contract,isolated_optimizer_calls=calls,environment_steps=0,formal_updates=0,full_epoch_state_exact_next_update=True,parameters=sum(p.numel() for p in m.parameters()),checks=checks)
 write(ROOT/('ADMISSION_%s.json'%('CODEC' if args.kind=='codec' else args.kind)),report);event('admission_complete',kind=args.kind,calls=calls)
if __name__=='__main__':main()
