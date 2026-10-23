import torch
from torch import nn
from torch.nn import functional as F
from .common import official,config,cpu
from .readout import readout
def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.ReLU(),nn.Linear(b,c))
def masked_mean(x,mask):
 mask=mask.bool();return x[mask].mean() if mask.any() else x.sum()*0
def huber(x,y):return F.smooth_l1_loss(x,y,reduction='none').mean(-1)
def grouped_huber(x,y):
 d=F.smooth_l1_loss(x,y,reduction='none');return (d[...,:3].mean(-1)+d[...,3:6].mean(-1)+d[...,6])/3
def selected(q):
 mu,sd,logits=q;k=logits.argmax(-1);ix=k[...,None,None].expand(*k.shape,1,7)
 return mu.gather(-2,ix).squeeze(-2),sd.gather(-2,ix).squeeze(-2),k
def distribution(q):
 m,s,l=q;return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=l),torch.distributions.Independent(torch.distributions.Normal(m,s),1))
def nll(q,a,mask):return masked_mean(-distribution(q).log_prob(a),mask)
def head():
 from libero.lifelong.models.policy_head import GMMHead
 return GMMHead(512,7,num_modes=config().policy.num_gmm_modes,min_std=1e-4)
def head_params(h,x):
 b,t,d=x.shape;m,s,l=h.forward_fn(x.reshape(b*t,d));return m.reshape(b,t,-1,7),s.reshape(b,t,-1,7),l.reshape(b,t,-1)
class Context(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,256),nn.ReLU(),nn.Linear(256,128))
 def forward(self,x):return self.net(x[:,-1,1:4].reshape(len(x),1536))
class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.context=Context();self.position=nn.Embedding(4,16);self.net=mlp(208,512,512);self.gmm=head()
 def forward(self,e,x):
  c=self.context(x)[:,None].expand(-1,4,-1);pos=self.position.weight[None].expand(len(e),-1,-1)
  return head_params(self.gmm,self.net(torch.cat([e,c,pos],-1)))
class Codec(nn.Module):
 def __init__(self,stats):
  super().__init__()
  for k,v in stats.items():self.register_buffer(k,v.clone())
  self.branches=nn.ModuleList([nn.Linear(7,64),nn.Linear(9,64),nn.Linear(64,64)])
  self.modality=nn.Parameter(torch.zeros(3,64));self.fuse=nn.Linear(64,256);self.query=nn.Parameter(torch.zeros(1,1,256))
  self.cross=nn.MultiheadAttention(256,4,dropout=0,batch_first=True);self.cross_norm=nn.LayerNorm(256)
  self.position=nn.Parameter(torch.zeros(1,4,256));self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,512,0,batch_first=True),2)
  self.effect=nn.Linear(256,64);self.response=mlp(64,128,9);self.visual=mlp(64,128,64);self.decoder=Decoder()
  raw=official();s={k[len('policy_head.'):]:v for k,v in raw.items() if k.startswith('policy_head.')};self.decoder.gmm.load_state_dict(s,strict=True)
 def forward(self,b):
  values=[(b[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std') for k in ('action','response','visual')]
  tokens=torch.stack([proj(v)*b[k+'_mask'][...,None]+self.modality[i] for i,(proj,v,k) in enumerate(zip(self.branches,values,('action','response','visual')))],2)
  bsz=len(tokens);tokens=self.fuse(tokens.reshape(bsz*4,3,64));q=self.query.expand(bsz*4,-1,-1)
  y=self.cross_norm(q+self.cross(q,tokens,tokens,need_weights=False)[0]).reshape(bsz,4,256)
  return self.effect(self.temporal(y+self.position))
 def reconstruction(self,e,b):
  return huber(self.response(e),(b['response']-self.response_mean)/self.response_std),huber(self.visual(e),(b['visual']-self.visual_mean)/self.visual_std)
class Policy(nn.Module):
 def __init__(self,arm,codec_state=None,statistics=None):
  super().__init__();self.arm=arm
  from libero.lifelong.models.modules.transformer_modules import OfficialGPT2Decoder
  # The original tensors are loaded exactly; additions have an independent RNG stream.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(410001);self.temporal_transformer=OfficialGPT2Decoder(512,6,8,.15,32,train=True)
  raw=official();sd={k[len('temporal_transformer.'):]:v for k,v in raw.items() if k.startswith('temporal_transformer.')};self.temporal_transformer.load_state_dict(sd,strict=True)
  assert all(torch.equal(v,self.temporal_transformer.state_dict()[k]) for k,v in sd.items())
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(410002)
   if arm=='B0':
    self.direct=mlp(512,512,4*512);self.gmm=head();self.gmm.load_state_dict({k[len('policy_head.'):]:v for k,v in raw.items() if k.startswith('policy_head.')},strict=True)
   else:
    self.predictor=mlp(512,512,256);self.decoder=Decoder();assert codec_state is not None and statistics is not None
    self.decoder.load_state_dict({k[len('decoder.'):]:v for k,v in codec_state.items() if k.startswith('decoder.')},strict=True)
    self.register_buffer('effect_mean',statistics['mean'].clone());self.register_buffer('effect_std',statistics['std'].clone())
 def forward(self,x):
  h,ifa=readout(self.temporal_transformer,x)
  if self.arm=='B0':return head_params(self.gmm,self.direct(h).reshape(len(h),4,512)),None,ifa
  z=self.predictor(h).reshape(len(h),4,64);e=z*self.effect_std+self.effect_mean;return self.decoder(e,x),z,ifa
 def decode_z(self,z,x):return self.decoder(z*self.effect_std+self.effect_mean,x)
def strict_policy_restore(model,payload,contract,codec_sha):
 assert payload['contract']==contract and payload['codec_sha256']==codec_sha
 expected=model.state_dict();assert set(expected)==set(payload['model'])
 assert all(v.shape==payload['model'][k].shape and v.dtype==payload['model'][k].dtype for k,v in expected.items())
 model.load_state_dict(payload['model'],strict=True)
 assert set(model.state_dict())==set(payload['model'])
 assert all(torch.equal(v.detach().cpu(),payload['model'][k].cpu()) for k,v in model.state_dict().items())
 return len(model.state_dict())
