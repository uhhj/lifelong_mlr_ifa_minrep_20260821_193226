import time, numpy as np, torch
from torch.nn.utils.stateless import functional_call
from .common import *
class Output(torch.nn.Module):
 def __init__(self,head):
  super().__init__();self.head=head
 def forward(self,z,actions=None):
  if actions is not None:return self.head.loss_fn(self.head(z),actions[:,None,:],reduction='none')
  means,stds,logits=self.head.forward_fn(z)
  return (logits.softmax(-1).unsqueeze(-1)*means).sum(-2)
class Engine:
 def __init__(self,head,z,actions,weight):
  self.module=Output(head).eval();self.names=[n for n,p in self.module.named_parameters()]
  self.shapes=[p.shape for p in self.module.parameters()];self.sizes=[p.numel() for p in self.module.parameters()]
  self.theta=torch.cat([p.detach().flatten() for p in self.module.parameters()]).clone()
  self.z=z;self.actions=actions;self.weight=weight;self.jvps=0;self.vjps=0;self.cvs=0
 def call(self,t,z,actions=None):
  chunks=t.split(self.sizes);params={n:v.view(s) for n,v,s in zip(self.names,chunks,self.shapes)}
  return functional_call(self.module,params,(z,actions))
 def output(self,t,role):return self.call(t,self.z[role])
 def losses(self,t,role):return self.call(t,self.z[role],self.actions[role])
 def grad(self,role):
  t=self.theta.detach().requires_grad_(True);v=self.losses(t,role).mean()
  return torch.autograd.grad(v,t)[0].detach()
 def jv(self,v,role='old_build'):
  self.jvps+=1
  return torch.autograd.functional.jvp(lambda t:self.output(t,role),self.theta,v,create_graph=False)[1].detach()
 def cv(self,v):
  self.cvs+=1;v=torch.as_tensor(v,dtype=self.theta.dtype,device=self.theta.device)
  j=self.jv(v);self.vjps+=1
  f=lambda t:self.output(t,'old_build')
  return torch.autograd.functional.vjp(f,self.theta,j*self.weight/len(j),create_graph=False)[1].detach()
 def trace(self):
  t=self.theta.detach().requires_grad_(True);out=self.output(t,'old_build');total=0.
  for i in range(out.shape[0]):
   for a in range(out.shape[1]):
    g=torch.autograd.grad(out[i,a],t,retain_graph=True)[0];self.vjps+=1
    total+=float(g.square().sum()*self.weight[a])/len(out)
   if (i+1)%32==0:print('TRACE',i+1,flush=True)
  return total

def load():
 from cb_ifa_experiment import gpu_runtime as env
 from libero.lifelong.models.brief import BriefPolicy
 from easydict import EasyDict
 torch.set_num_threads(1);torch.manual_seed(2026092302);np.random.seed(2026092302)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 c=read(OUT/'EXPERIMENT_CONTEXT.json');cfg=read(Path(c['checkpoint']).parent/'config.json')
 cfg.update(aug=False,device='cuda:0',pretrain=c['checkpoint']);cfg=EasyDict(cfg)
 model=BriefPolicy(cfg,cfg.shape_meta)
 state=torch.load(c['checkpoint'],map_location='cpu')['state_dict'];state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
 model.load_state_dict(state,strict=True);model=model.cuda().eval()
 split=read(OUT/'DATA_SPLIT.json');z={};actions={};parity=[]
 for role,rows in split['roles'].items():
  zs=[];aa=[]
  for start in range(0,len(rows),16):
   data=[torch.load(ROOT/r['origin'],map_location='cpu') for r in rows[start:start+16]]
   h=torch.cat([d['feature'] for d in data],0).cuda();a=torch.cat([d['actions'] for d in data],0).cuda()
   with torch.no_grad():
    terminal=model.temporal_encode(h)[:,-1,:];zs.append(terminal);aa.append(a[:,-1,:])
    if start==0:
     # Official training dispatch, all children remain eval so no dropout.
     model.training=True;dist,feat=model(h)
     model.training=False
     parity.append(float((feat-terminal).abs().max()))
  z[role]=torch.cat(zs).detach();actions[role]=torch.cat(aa).detach()
 # Fixed per-group training/build statistics. Equal weight to three control groups.
 a=actions['old_build'].double();groups=[(0,3),(3,6),(6,7)];w=[];stats=[]
 for lo,hi in groups:
  raw=float(a[:,lo:hi].var(0,unbiased=False).mean().sqrt());s=max(raw,.05)
  w.extend([1/((hi-lo)*s*s)]*(hi-lo));stats.append(dict(indices=list(range(lo,hi)),rms_std=raw,used_scale=s))
 metric=dict(groups=stats,diagonal=w,source='old_build terminal expert actions only',scale_floor=.05,definition='equal group weight / within-group variance; no probe statistics')
 if (OUT/'ACTION_METRIC.json').exists():assert read(OUT/'ACTION_METRIC.json')==metric
 else:write(OUT/'ACTION_METRIC.json',metric)
 head=model.policy_head;out=Output(head)
 with torch.no_grad():
  mean32=out(z['old_probe']).double();loss32=out(z['old_probe'],actions['old_probe']).double()
 head=head.double();z={k:v.double() for k,v in z.items()};actions={k:v.double() for k,v in actions.items()}
 engine=Engine(head,z,actions,torch.tensor(w,device='cuda',dtype=torch.float64))
 with torch.no_grad():
  mean64=engine.output(engine.theta,'old_probe');loss64=engine.losses(engine.theta,'old_probe')
 checks=dict(strict_checkpoint_load=True,eval_decoder=True,eval_head=True,official_terminal_feature_max_difference=max(parity),head_float64_vs_native_mean_max_error=float((mean64-mean32).abs().max()),head_float64_vs_native_nll_max_error=float((loss64-loss32).abs().max()),dtype='decoder float32; diagnostic head float64 for every branch',head_parameters=engine.theta.numel(),optimizer_steps_executed=0)
 assert checks['official_terminal_feature_max_difference']==0
 assert checks['head_float64_vs_native_mean_max_error']<1e-4 and checks['head_float64_vs_native_nll_max_error']<1e-2
 write(OUT/'MODEL_INTERFACE_CHECK.json',checks)
 del model,state;torch.cuda.empty_cache()
 return engine
