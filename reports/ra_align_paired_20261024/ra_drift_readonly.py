"""Strict frozen ALIGN CPU inference on identical retained Task8 windows. No writes."""
import os,sys,ast,json,hashlib,time,math
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.nice(10)
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import robomimic.utils.tensor_utils as TensorUtils
from transformers import GPT2Config,GPT2Model
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False);torch.manual_seed(710041)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def emit(d):print(json.dumps(d,allow_nan=False,separators=(',',':')),flush=True)
def select(q,k=None):
 mu,sd,logits=q;k=logits.argmax(-1) if k is None else k;i=torch.arange(len(k));return mu[i,k],sd[i,k],k
def gh(mu,a):
 x=F.smooth_l1_loss(mu,a,reduction='none');return (x[:,:3].mean(-1)+x[:,3:6].mean(-1)+x[:,6])/3
def scores(q,a):
 mu,sd,l=[v.double() for v in q];m,s,k=select((mu,sd,l));lp=D.Normal(mu,sd).log_prob(a[:,None,:].double()).sum(-1)
 mix=-torch.logsumexp(l.log_softmax(-1)+lp,-1);sel=-lp.gather(-1,k[:,None]).squeeze(-1)
 dist=D.MixtureSameFamily(D.Categorical(logits=l),D.Independent(D.Normal(mu,sd),1))
 assert torch.allclose(mix,-dist.log_prob(a.double()),atol=1e-8,rtol=1e-10)
 return mix,sel,gh(m,a.double())
start=time.monotonic();cpu_start=time.process_time();totalcalls=0;totals=[]
emit({'kind':'start','status':'READONLY_CPU_DIAGNOSTIC_STARTED','time':time.time(),'expected_windows':1356,'expected_calls':174})
for entry in SPEC['seeds']:
 seed=entry['seed'];O=Path(entry['output']);S=Path(entry['source']);RA=Path(entry['ra_output']);tic=time.monotonic();tc=time.process_time()
 assert sha(O/'CONTRACT.json')==entry['contract_sha256'];contract=js(O/'CONTRACT.json')
 def verify_source():
  assert all(sha(S/k)==h for k,h in contract['source_sha256'].items())
  for k,h in entry['metadata'].items():assert sha(O/k)==h
 verify_source()
 assert sha(RA/'CONTRACT.json')==entry['ra_contract_sha256']
 assert js(RA/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
 assert [(r['destination'],r['sha256']) for r in js(RA/'ra_coreset/storage/task8.json')['selected']]==[(r['destination'],r['sha256']) for r in js(O/'ra_coreset/storage/task8.json')['selected']]
 assert (O/'pipeline.exit').read_text().strip()==(O/'supervisor.exit').read_text().strip()=='0'
 assert js(O/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
 def exact_class(name,path):
  node=next(n for n in ast.parse((S/path).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(S/path),'exec'),globals())
 exact_class('GMMHead','libero/lifelong/models/policy_head.py')
 exact_class('OfficialGPT2Decoder','libero/lifelong/models/modules/transformer_modules.py')
 brief=next(n for n in ast.parse((S/'libero/lifelong/models/brief.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BriefPolicy')
 meth=next(n for n in brief.body if isinstance(n,ast.FunctionDef) and n.name=='temporal_encode')
 exec(compile(ast.Module(body=[meth],type_ignores=[]),str(S/'libero/lifelong/models/brief.py'),'exec'),globals())
 class Policy(nn.Module):
  temporal_encode=temporal_encode
  def __init__(self):
   super().__init__();self.temporal_transformer=OfficialGPT2Decoder(512,6,8,.15,32,train=True);self.policy_head=GMMHead(512,7,num_modes=5,min_std=1e-4)
  def forward(self,x):return self.policy_head.forward_fn(self.temporal_encode(x)[:,-1])
 models=[];states=[];cpmeta={}
 for stage in (8,9):
  cp=RA/'ra_coreset/models'/f'mlrifa_{stage}_model_ep99.pth';assert sha(cp)==entry['ra_checkpoints'][str(stage)]
  raw=torch.load(cp,map_location='cpu')['state_dict'];sd={(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()}
  p=Policy().eval().requires_grad_(False);expected=p.state_dict();assert len(sd)==len(expected)==100 and set(sd)==set(expected)
  assert all(sd[k].shape==v.shape and sd[k].dtype==v.dtype for k,v in expected.items())
  p.load_state_dict(sd,strict=True);assert all(torch.equal(v,sd[k]) for k,v in p.state_dict().items())
  assert all(not m.training for m in p.modules()) and not p.policy_head.low_eval_noise
  models.append(p);states.append(sd);cpmeta[str(stage)]={'sha256':sha(cp),'strict_keys':100,'low_eval_noise':p.policy_head.low_eval_noise}
 side=O/'effects/memory/task8.pth';sidehash=sha(side);assert sidehash==js(side.with_suffix('.json'))['sha256']
 labels=torch.load(side,map_location='cpu');storage=js(O/'ra_coreset/storage/task8.json')['selected'];manifest=js(O/'ra_coreset/dataset_manifests/task9.json')['rows']
 stage9={Path(r['path']).name:r for r in manifest if r['task']==8};assert len(storage)==len(labels)==len(stage9)==452
 assert {r['destination'] for r in storage}==set(labels)==set(stage9)
 rows=[];calls=0
 for offset in range(0,len(storage),16):
  members=storage[offset:offset+16];features=[];targets=[]
  for member in members:
   name=member['destination'];r=labels[name];md=stage9[name];fp=O/'ra_coreset/buffer'/name
   assert sha(RA/'ra_coreset/buffer'/name)==sha(fp)==member['sha256']==r['sample_sha256']==md['sha256']
   assert r['bc_valid']==md['bc_valid'] and r['task']==md['task']==8 and (r['demo'],r['u'])==(md['demo'],md['u'])
   if r['bc_valid']:assert r['identity_unique'] and r['full_window_valid'] and r['next_action_index']==r['u']+1
   features.append(torch.load(fp,map_location='cpu')['feature'].squeeze(0));targets.append(r['next_action'])
  x=torch.stack(features);a=torch.stack(targets);assert x.shape[1:]==(8,4,512) and a.shape[1:]==(7,)
  q0=models[0](x);q1=models[1](x);calls+=2;totalcalls+=2
  assert all(torch.isfinite(v).all() for q in (q0,q1) for v in q) and all((q[1]>0).all() for q in (q0,q1))
  q0=tuple(v.double() for v in q0);q1=tuple(v.double() for v in q1)
  mu0,sd0,k0=select(q0);mu1,sd1,k1=select(q1);mus,sds,_=select(q1,k0)
  dmu=(mu1-mu0).square();dsd=(sd1-sd0).square();same=(mus-mu0).square()+(sds-sd0).square()
  allcost=(q1[0]-mu0[:,None]).square().sum(-1)+(q1[1]-sd0[:,None]).square().sum(-1);best,bestk=allcost.min(-1)
  sc0=scores(q0,a);sc1=scores(q1,a);pr0=q0[2].softmax(-1);pr1=q1[2].softmax(-1);top=pr0.topk(2,-1)
  for j,member in enumerate(members):
   name=member['destination'];r=labels[name];valid=r['bc_valid']
   row={'sample':name,'sample_sha256':member['sha256'],'demo':r['demo'],'u':r['u'],'bc_valid':bool(valid),
    'teacher_mode':int(k0[j]),'student_mode':int(k1[j]),'mode_switched':bool(k0[j]!=k1[j]),
    'teacher_probability_margin':float(top.values[j,0]-top.values[j,1]),
    'teacher_logit_margin':float(q0[2][j].topk(2).values.diff().abs()[0]),
    'teacher_pmax':float(top.values[j,0]),'student_pmax':float(pr1[j].max()),
    'gripper_strict_sign_flip':bool(mu0[j,6]*mu1[j,6]<0),'gripper_sign_changed':bool(torch.sign(mu0[j,6])!=torch.sign(mu1[j,6])),
    'mu_delta_squared_by_dim':dmu[j].tolist(),'sigma_delta_squared_by_dim':dsd[j].tolist(),'same_teacher_index_w2_squared_by_dim':same[j].tolist(),
    'selected_w2_squared':float((dmu[j]+dsd[j]).sum()),'same_teacher_index_w2_squared':float(same[j].sum()),
    'nearest_student_component_w2_squared':float(best[j]),'nearest_student_component':int(bestk[j]),
    'teacher_gripper_sigma':float(sd0[j,6]),'student_gripper_sigma':float(sd1[j,6]),
    'gripper_positive_probability_delta':float((.5*(1+torch.erf(mu1[j,6]/sd1[j,6]/math.sqrt(2))))-(.5*(1+torch.erf(mu0[j,6]/sd0[j,6]/math.sqrt(2)))))}
   for tag,sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
    row[tag+'_mean_l2_delta']=float(dmu[j,sl].sum().sqrt());row[tag+'_w2_squared']=float((dmu[j,sl]+dsd[j,sl]).sum())
   row['expert']={}
   if valid:
    row['next_action_sha256']=hashlib.sha256(a[j].numpy().tobytes()).hexdigest()
    for name_,v0,v1 in zip(('mixture_nll','selected_nll','grouped_huber'),sc0,sc1):row['expert'][name_]={'teacher':float(v0[j]),'student':float(v1[j]),'delta':float(v1[j]-v0[j])}
   rows.append(row)
  if (offset//16)%8==7:emit({'kind':'progress','seed':seed,'forward_calls':calls,'windows':len(rows)})
 verify_source();assert sha(side)==sidehash
 for p,sd,stage in zip(models,states,(8,9)):
  assert all(torch.equal(v,sd[k]) for k,v in p.state_dict().items())
  assert sha(RA/'ra_coreset/models'/f'mlrifa_{stage}_model_ep99.pth')==entry['ra_checkpoints'][str(stage)]
 assert all(sha(O/'ra_coreset/buffer'/r['destination'])==r['sha256'] for r in storage)
 assert calls==58 and len(rows)==452 and not torch.cuda.is_initialized()
 evidence={'kind':'seed','seed':seed,'status':'PASS_CPU_READONLY_PAIRED_DIAGNOSTIC','windows':len(rows),'valid_expert_windows':sum(r['bc_valid'] for r in rows),'forward_calls':calls,'checkpoints':cpmeta,'contract_sha256':entry['ra_contract_sha256'],'common_align_contract_sha256':entry['contract_sha256'],'label_sidecar_sha256':sidehash,'metadata_sha256':entry['metadata'],'source_sha256':{n:sha(S/n) for n in ('libero/lifelong/models/brief.py','libero/lifelong/models/policy_head.py','libero/lifelong/models/modules/transformer_modules.py')},'unchanged_after':True,'rows':rows,'cpu_seconds':time.process_time()-tc,'wall_seconds':time.monotonic()-tic}
 emit(evidence);totals.append({k:v for k,v in evidence.items() if k not in ('rows','metadata_sha256','source_sha256')});del models,states,p,sd,raw,labels,rows
assert totalcalls==174 and not torch.cuda.is_initialized()
emit({'kind':'complete','status':'PASS_CPU_READONLY_NOT_CAUSAL_CLOSED_LOOP_TEST','windows':1356,'policy_windows':2712,'forward_calls':totalcalls,'seeds':totals,'cpu_seconds':time.process_time()-cpu_start,'wall_seconds':time.monotonic()-start,'optimizer_updates':0,'backwards':0,'environment_steps':0,'server_artifact_writes':0,'cuda_initialized':False,'cpu_threads':1})
