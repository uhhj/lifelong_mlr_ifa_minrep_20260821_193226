"""Bounded offline inference only. Never constructs environments or updates weights."""
import os,sys,json,time,hashlib,copy
from pathlib import Path
import torch
torch.cuda.set_per_process_memory_fraction(0.24)
torch.backends.cudnn.benchmark=True
from coreset_exp.model import cfg,data_setup,complete,strict_downstream
from coreset_exp.common import O,sha
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import control_seed

def clone(d):return {k:({kk:vv.clone() for kk,vv in v.items()} if isinstance(v,dict) else v.clone()) for k,v in d.items()}
def shamodel(p):
 h=hashlib.sha256()
 for k,v in p.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def compare(a,b):return dict(equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()),different=int(torch.count_nonzero(a!=b)),elements=a.numel())
def main():
 control_seed(809110);c=cfg('eval');b,_,shape=data_setup(c,False);c.shape_meta=shape
 p=complete(c,shape).eval().requires_grad_(False);before=shamodel(p)
 task=6;ds,_=get_dataset(str(Path(c.folder)/b.get_task_demonstration(task)),c.data.obs.modality,False,seq_len=8,demo_limit=10,train=False)
 indices=[min(i*31,len(ds)-1) for i in range(10)]
 rows=[ds[i] for i in indices];obs={k:torch.stack([torch.as_tensor(r['obs'][k]) for r in rows]).cuda() for k in shape['all_shapes']}
 instr,mask=b.get_task_emb(task);base=dict(task_emb=instr.repeat(10,1).cuda(),masks=mask.repeat(10,1).cuda())
 def data(start,end):return dict(obs={k:v[:,start:end].contiguous() for k,v in obs.items()},**base)
 def enc(d):return p.spatial_encode(p.preprocess_input(clone(d),train_mode=False))
 def timed(fn):
  torch.cuda.synchronize();t=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-t
 results=[];times=[]
 with torch.no_grad():
  frames=[]
  for i in range(8):
   x,t=timed(lambda:enc(data(i,i+1)));frames.append(x);times.append(t)
  for n in range(1,9):
   whole,t=timed(lambda:enc(data(0,n)));cached=torch.cat(frames[:n],1)
   repeat,_=timed(lambda:enc(data(0,n)))
   results.append(dict(length=n,feature=compare(whole,cached),original_repeat=compare(whole,repeat),full_seconds=t))
  policies=[]
  for stage in range(6,10):
   cp=O/'random/models'/f'mlrifa_{stage}_model_ep99.pth';strict_downstream(p,cp)
   rng=torch.cuda.get_rng_state().clone();torch.cuda.set_rng_state(rng)
   y,t1=timed(lambda:p.temporal_encode(whole)[:,-1]);d1=p.policy_head(y,train=False);a1=d1.sample();r1=torch.cuda.get_rng_state().clone()
   torch.cuda.set_rng_state(rng);z,t2=timed(lambda:p.temporal_encode(cached)[:,-1]);d2=p.policy_head(z,train=False);a2=d2.sample();r2=torch.cuda.get_rng_state().clone()
   policies.append(dict(stage=stage,checkpoint_sha256=sha(cp),embedding=compare(y,z),sampled_action=compare(a1,a2),rng_equal=bool(torch.equal(r1,r2)),temporal_seconds=[t1,t2]))
  # Isolate the full-window vision and language costs with synchronized calls.
  d=p.preprocess_input(clone(data(0,8)),train_mode=False)
  lang,lt=timed(lambda:p.language_encoder(d['task_emb'],d['masks'].squeeze(1)))
  vision=[]
  for key in p.rgbkey:
   imgs=d['obs'][key];_,tt=timed(lambda:p.image_encoders(imgs.reshape(80,*imgs.shape[2:]),langs=None));vision.append(dict(camera=key,seconds=tt))
  # Cached language alone must retain the same call shape and exact result.
  repeatlang,_=timed(lambda:p.language_encoder(d['task_emb'],d['masks'].squeeze(1)))
  language=compare(lang,repeatlang)
 print('RESULT_JSON='+json.dumps(dict(status='PASS_EXACT_CACHE' if all(x['feature']['equal'] for x in results) and all(x['sampled_action']['equal'] for x in policies) else 'REJECT_FRAME_CACHE_NOT_BITWISE_EQUAL',suite='OBJECT',cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,batch=10,lengths=results,policy=policies,single_frame_seconds=times,language_seconds=lt,language_repeat=language,vision_seconds=vision,initial_model_sha256=before,peak_allocated=torch.cuda.max_memory_allocated(),new_optimizer=0,new_environment=0,production_writes=0,input_source='Existing task6 demonstration windows; indices recorded, payload not exported',indices=indices,shared_gpu=True,timing_not_uncontended_speedup=True)))
if __name__=='__main__':main()
