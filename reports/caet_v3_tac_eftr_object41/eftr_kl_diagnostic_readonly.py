"""Zero-update CPU diagnostic of one atomically read completed-epoch snapshot.

No experiment files written, no optimizer constructed, no environments or GPU.
Only scalar divergences, identities, hashes and summaries leave the server.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.nice(10)
import ast, io, json, hashlib, math, time, sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from transformers import GPT2Config, GPT2Model
import robomimic.utils.tensor_utils as TensorUtils
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(710041)
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
S=Q/'EFTR/source'
O=Path('/data/mlr_ifa/experiments/CAETv3_EFTR_OBJECT_seed41_20261022_collate01')
tic=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
contract=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
assert all(sha(Path(p))==h for p,h in contract['files'].items())
source_hashes={n:sha(S/n) for n in ['libero/lifelong/models/brief.py','libero/lifelong/models/policy_head.py','libero/lifelong/models/modules/transformer_modules.py','caet_v3_exp/model.py','caet_v3_exp/observe.py','caet_v3_exp/eftr_optimizer.py']}
def exact_class(name,filename):
 tree=ast.parse((S/filename).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(S/filename),'exec'),globals())
exact_class('GMMHead','libero/lifelong/models/policy_head.py')
exact_class('OfficialGPT2Decoder','libero/lifelong/models/modules/transformer_modules.py')
brief=ast.parse((S/'libero/lifelong/models/brief.py').read_text())
klass=next(n for n in brief.body if isinstance(n,ast.ClassDef) and n.name=='BriefPolicy')
method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='temporal_encode')
exec(compile(ast.Module(body=[method],type_ignores=[]),str(S/'libero/lifelong/models/brief.py'),'exec'),globals())
class ReadOnlyDownstream(nn.Module):
 temporal_encode=temporal_encode
 def __init__(self):
  super().__init__()
  self.temporal_transformer=OfficialGPT2Decoder(embed_size=512,num_layers=6,num_heads=8,dropout=.15,max_position_embeddings=32,train=True)
  self.policy_head=GMMHead(input_size=512,output_size=7,num_modes=5,min_std=1e-4)
 def parameters_for(self,x):
  return self.policy_head.forward_fn(self.temporal_encode(x)[:,-1])
def load_model(sd):
 model=ReadOnlyDownstream().eval().requires_grad_(False)
 sd={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in sd.items()}
 assert len(sd)==100 and set(sd)==set(model.state_dict())
 model.load_state_dict(sd,strict=True)
 return model
# Atomic writer uses os.replace: one open file descriptor pins one whole checkpoint.
cp=O/'ra_coreset/recovery/latest.pth'
with cp.open('rb') as f:
 before=os.fstat(f.fileno());raw=f.read();after=os.fstat(f.fileno())
assert before.st_ino==after.st_ino and before.st_size==after.st_size==len(raw)
cp_sha=hashlib.sha256(raw).hexdigest();checkpoint=torch.load(io.BytesIO(raw),map_location='cpu');del raw
assert checkpoint['task']==7 and 1<=checkpoint['next_epoch']<=100
assert checkpoint['contract_sha256']==sha(O/'CONTRACT.json')
epoch=checkpoint['next_epoch']-1;step=checkpoint['state']['step']
assert step==15700+227*(epoch+1)
model=load_model(checkpoint['state']['model']);del checkpoint
teacher_path=O/'teachers/task6.pth';meta=json.loads(teacher_path.with_suffix('.json').read_text())
assert sha(teacher_path)==meta['sha256']
teachers=torch.load(teacher_path,map_location='cpu')
teacher_cp=O/'ra_coreset/models/mlrifa_6_model_ep99.pth'
assert sha(teacher_cp)==meta['checkpoint_sha256']
teacher_model=load_model(torch.load(teacher_cp,map_location='cpu')['state_dict'])
storage_path=O/'ra_coreset/storage/task6.json';assert sha(storage_path)==meta['storage_sha256']
storage=json.loads(storage_path.read_text())['selected'];assert len(storage)==len(teachers)==781
rows=[];replay={'means_max_abs':0.,'std_max_abs':0.,'logits_max_abs':0.,'mode_mismatches':0};counter=0
def select(values,modes=None):
 m,s,l=values;k=l.argmax(-1) if modes is None else modes;i=torch.arange(len(k));return m[i,k].double(),s[i,k].double(),k
def decomposition(ref,now):
 m0,s0,k0=ref;m,s,k=now
 mean=(m0-m).square()/(2*s.square())
 variance=torch.log(s/s0)+s0.square()/(2*s.square())-.5
 return mean,variance,mean+variance
with torch.no_grad():
 for start in range(0,len(storage),16):
  batch=storage[start:start+16];xs=[];ts=[]
  for b in batch:
   p=O/'ra_coreset/buffer'/b['destination'];assert sha(p)==b['sha256']==teachers[b['destination']]['sample_sha256']
   xs.append(torch.load(p,map_location='cpu')['feature'].squeeze(0));ts.append(teachers[b['destination']])
  x=torch.stack(xs);assert x.shape[1:]==(8,4,512)
  tv=tuple(torch.stack([t[n] for t in ts]) for n in ('means','std','logits'))
  rebuilt=teacher_model.parameters_for(x);cur=model.parameters_for(x);counter+=2
  for key,left,right in zip(['means','std','logits'],tv,rebuilt):replay[key+'_max_abs']=max(replay[key+'_max_abs'],float((left-right).abs().max()))
  replay['mode_mismatches']+=int((tv[2].argmax(-1)!=rebuilt[2].argmax(-1)).sum())
  ref=select(tv);actual=select(cur);same=select(cur,ref[2]);dm,dv,dk=decomposition(ref,actual);sm,sv,sk=decomposition(ref,same)
  # Relative-scale probes are arithmetic only; no loss, model or optimizer changes.
  floor_results={}
  for floor in (.001,.01,.05):
   r=(ref[0],ref[1].clone(),ref[2]);v=(actual[0],actual[1].clone(),actual[2]);r[1][:,6].clamp_(min=floor);v[1][:,6].clamp_(min=floor)
   floor_results[str(floor)]=decomposition(r,v)[2].sum(-1)/7+.1*(ref[2]!=actual[2]).double()
  # Gaussian sign probabilities: only a distribution diagnostic, not closed-loop control.
  pT=.5*(1+torch.erf(ref[0][:,6]/(ref[1][:,6]*math.sqrt(2))))
  pC=.5*(1+torch.erf(actual[0][:,6]/(actual[1][:,6]*math.sqrt(2))))
  for j,b in enumerate(batch):
   switched=bool(ref[2][j]!=actual[2][j]);r=float(dk[j].sum()/7)+.1*switched
   rows.append(dict(sample=b['destination'],sample_sha256=b['sha256'],teacher_mode=int(ref[2][j]),current_mode=int(actual[2][j]),switched=switched,
    kl_by_dim=dk[j].tolist(),mean_term_by_dim=dm[j].tolist(),variance_term_by_dim=dv[j].tolist(),same_mode_kl_by_dim=sk[j].tolist(),
    arm_contribution_R=float(dk[j,:6].sum()/7),gripper_contribution_R=float(dk[j,6]/7),explicit_switch_contribution_R=.1*switched,R=r,
    actual_minus_same_mode_R=float((dk[j]-sk[j]).sum()/7),arm_only_R=float(dk[j,:6].sum()/7)+.1*switched,
    gripper_std_teacher=float(ref[1][j,6]),gripper_std_current=float(actual[1][j,6]),gripper_mean_abs_delta=float((ref[0][j,6]-actual[0][j,6]).abs()),
    gripper_mean_sign_changed=bool(torch.sign(ref[0][j,6])!=torch.sign(actual[0][j,6])),gripper_sign_probability_abs_delta=float((pT[j]-pC[j]).abs()),
    arithmetic_floor_R={k:float(v[j]) for k,v in floor_results.items()}))
assert replay['mode_mismatches']==0 and replay['means_max_abs']<1e-4 and replay['std_max_abs']<1e-4 and replay['logits_max_abs']<1e-3,replay
# Original scores only, no candidate reconstruction or extra optimizer calls.
score_rows=[]
for ep in range(epoch+1):
 p=O/'ra_coreset/auxiliary'/('task7_epoch%02d.json'%ep)
 for row in json.loads(p.read_text()):
  score_rows.append({k:row[k] for k in ['step','scores','selected','scale','budget_unmet','optimizer_trials','committed_updates','retention_loss','mode_loss']})
constraint=json.loads((O/'ra_coreset/constraints'/('task7_epoch%02d.json'%epoch)).read_text())
assert len(score_rows)==227*(epoch+1)
assert all(sha(Path(p))==h for p,h in contract['files'].items())
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='ZERO_TRAINING_DIAGNOSTIC_EVIDENCE',observed_unix=time.time(),snapshot=dict(path=str(cp),sha256=cp_sha,task=7,epoch=epoch,step=step,bytes=before.st_size,atomic_fd=True),
 source_hashes=source_hashes,actual_head='GMMHead',actual_low_eval_noise=model.policy_head.low_eval_noise,strict_model_keys=100,teacher_meta=meta,teacher_replay=replay,
 constraint=constraint,rows=rows,candidate_scores=score_rows,model_forward_calls=counter,optimizer_updates=0,backwards=0,environment_steps=0,cuda_initialized=False,server_writes=0,cpu_threads=2,seconds=time.monotonic()-tic),allow_nan=False))
