"""Same legal RA samples for every historical reference. Never future models."""
import json
from pathlib import Path
import torch
from coreset_exp.common import O,sha,write,rng,restore_rng
from .variant import feature

def nll(policy,x,actions,head=None):
 z=policy.temporal_encode(x.squeeze(1) if x.ndim==5 else x)[:,-1]
 distribution=(policy.policy_head if head is None else head)(z)
 return -distribution.log_prob(actions)

def groups(ds,t):
 candidates={}
 for i,r in enumerate(ds.rows):
  if r['valid'] and r['task']<t:
   key=str(r['task'])+'|'+r['group'];candidates.setdefault(key,[]).append(i)
 return {k:ii for k,ii in candidates.items() if len({ds.rows[i]['demo'] for i in ii})>=3 and len({(ds.rows[i]['demo'],ds.rows[i]['u']) for i in ii})>=12}

def score(policy,ds,mapping,head):
 result={};modes={n:m.training for n,m in policy.named_modules()};state=rng()
 try:
  policy.eval()
  with torch.no_grad():
   for key,indices in mapping.items():
    total=0.
    for begin in range(0,len(indices),64):
     ix=indices[begin:begin+64];x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ix]).cuda();actions=torch.stack([ds.rows[i]['next_action'] for i in ix]).cuda()
     values=nll(policy,feature(x),actions,head);assert torch.isfinite(values).all();total+=float(values.double().sum())
    result[key]=total/len(indices)
 finally:
  for n,m in policy.named_modules():m.training=modes[n]
  restore_rng(state)
 return result

def build(policy,ds,t,head):
 saved_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
 mapping=groups(ds,t);saved={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()};table={k:[] for k in mapping};cps={}
 try:
  for s in range(6,t):
   chosen={k:v for k,v in mapping.items() if int(k.split('|')[0])<=s}
   cp=O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%s);raw=torch.load(cp,map_location='cpu');assert raw['task']==s and raw['epoch_index']==99
   policy.load_state_dict({k.removeprefix('module.') if hasattr(k,'removeprefix') else k[7:] if k.startswith('module.') else k:v for k,v in raw['state_dict'].items()},strict=True)
   hp=O/'ra_coreset/aux_heads'/('task%d_ep99.pth'%s);payload=torch.load(hp,map_location='cpu');assert payload['task']==s and payload['epoch']==99
   head.load_state_dict(payload['state']['model'],strict=True)
   cps[str(s)]=dict(policy_sha256=sha(cp),auxiliary_head_sha256=sha(hp))
   for key,v in score(policy,ds,chosen,head).items():table[key].append(dict(stage=s,nll=v))
 finally:
  policy.load_state_dict(saved,strict=True);head.load_state_dict(saved_head,strict=True)
 result={k:dict(bound=min(v,key=lambda x:(x['nll'],x['stage']))['nll'],reference_stage=min(v,key=lambda x:(x['nll'],x['stage']))['stage'],candidates=v,count=len(mapping[k]),members=[dict(sample=ds.rows[i]['sample'],demo=ds.rows[i]['demo'],u=ds.rows[i]['u'],sha256=ds.rows[i]['sample_sha256']) for i in mapping[k]]) for k,v in table.items()}
 write(O/'ra_coreset/functional_references'/('task%d.json'%t),dict(groups=result,checkpoints=cps,epsilon=.05,target="TRAINING_ONLY_AUXILIARY_NEXT_ACTION_HEAD"))
 return mapping,{k:v['bound'] for k,v in result.items()}

def dual(mu,values,bounds):
 return {k:max(0.,min(1.,mu[k]+.1*(values[k]-bounds[k]-.05))) for k in bounds}
