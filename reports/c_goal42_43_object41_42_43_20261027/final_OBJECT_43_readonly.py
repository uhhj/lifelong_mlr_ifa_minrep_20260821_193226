"""External CPU-only final audit and fixed expert-cost diagnostic. No server writes."""
import os,sys,ast,json,hashlib,time,contextlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1/OBJECT_43');O=Path('/data/mlr_ifa/experiments/ALIGN_OBJECT_seed43_20261022');S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_43/source')
sys.path.insert(0,str(Q))
from layout import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import robomimic.utils.tensor_utils as TensorUtils
from transformers import GPT2Config,GPT2Model
torch.set_num_threads(1);torch.set_grad_enabled(False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def verify():
 c=js(Q/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items());assert all(sha(S/n)==h for n,h in c['parent_source'].items())
verify();assert (Q.parent/'controller.exit').read_text()=='0'
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  assert int(p.name)==os.getpid() or str(Q.parent) not in cmd,('Residual',p.name)
 except OSError:pass
def exact_class(name,path):
 n=next(x for x in ast.parse((S/path).read_text()).body if isinstance(x,ast.ClassDef) and x.name==name);exec(compile(ast.Module(body=[n],type_ignores=[]),str(S/path),'exec'),globals())
exact_class('OfficialGPT2Decoder','libero/lifelong/models/modules/transformer_modules.py');exact_class('GMMHead','libero/lifelong/models/policy_head.py')
brief=next(x for x in ast.parse((S/'libero/lifelong/models/brief.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='BriefPolicy');method=next(x for x in brief.body if isinstance(x,ast.FunctionDef) and x.name=='temporal_encode');exec(compile(ast.Module(body=[method],type_ignores=[]),'native_temporal_encode','exec'),globals())
class Policy(nn.Module):
 temporal_encode=temporal_encode
 def __init__(self):
  super().__init__();self.temporal_transformer=OfficialGPT2Decoder(512,6,8,.15,32,train=True);self.policy_head=GMMHead(512,7,num_modes=5,min_std=1e-4)
 def forward(self,x):return self.policy_head.forward_fn(self.temporal_encode(x)[:,-1])
def model(path):
 raw=torch.load(path,map_location='cpu')['state_dict'];sd={(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()};p=Policy().eval().requires_grad_(False)
 assert len(sd)==len(p.state_dict())==100 and all(sd[k].shape==v.shape and sd[k].dtype==v.dtype for k,v in p.state_dict().items());p.load_state_dict(sd,strict=True);assert all(torch.equal(v,sd[k]) for k,v in p.state_dict().items());return p
# Appended to external native CPU policy loader by build_final_wrapper.py.
assert (Q/'train_C.exit').read_text()=='0' and (Q/'evaluate.exit').read_text()=='0'
contract=js(Q/'CONTRACT.json');assert sha(contract['official_initialization']['path'])==contract['official_initialization']['sha256']
parent_binding=js(Q/'PARENT_BINDING.json')
assert all(sha(p)==h for p,h in parent_binding['files'].items())
initial_record=js(Q/'INITIAL_C.json');official_record=js(O/'ra_coreset/INITIAL_STATE.json')
assert initial_record['tensors']==official_record['tensors']
assert all(initial_record['rng'][k]==official_record['rng'][k] for k in initial_record['rng'] if k!='cuda')
manifests={};stage_labels={}
for stage in range(6,10):
 manifest=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(manifest)==POP[stage] and all(sha(r['path'])==r['sha256'] for r in manifest)
 labels={}
 for t in range(6,stage):
  p=O/('effects/memory/task%d.pth'%t);memory=torch.load(p,map_location='cpu');storage=js(O/('ra_coreset/storage/task%d.json'%t))['selected']
  md={Path(r['path']).name:r for r in manifest if r['task']==t};assert set(md)==set(memory)=={r['destination'] for r in storage}
  for r in storage:assert r['sha256']==md[r['destination']]['sha256']==memory[r['destination']]['sample_sha256']
  labels.update({str(O/'ra_coreset/buffer'/n):v for n,v in memory.items()})
 p=O/('effects/current/task%d/labels.pth'%stage);labels.update({str(O/'features'/str(stage)/n):v for n,v in torch.load(p,map_location='cpu').items()})
 aligned=[]
 for m in manifest:
  r=labels[m['path']];assert r['sample_sha256']==m['sha256'] and all(r[k]==m[k] for k in ('task','demo','u','bc_valid'))
  if r['bc_valid']:assert r['identity_unique'] and r['full_window_valid'] and r['next_action_index']==r['u']+1
  aligned.append(r)
 manifests[stage]=manifest;stage_labels[stage]=aligned
from directional_prototype import directional_anchor
strict={};diags={};teacher_audits={};calls=0;windows=0;tic=time.time();task6={};historical_task6={}
for arm in ARMS:
 strict[arm]={};hist=js(Q/('EPOCHS_'+arm+'.json'));assert len(hist)==400
 for i,h in enumerate(hist):
  stage=6+i//100;order=torch.randperm(POP[stage],generator=torch.Generator().manual_seed(i%100)).numpy().astype(np.int64)
  assert hashlib.sha256(order.tobytes()).hexdigest()==h['sampler_sha256']
 for stage in range(6,10):
  cp=model_path(arm,stage);p=model(cp);meta=js(Q/('STAGE_%s_%d.json'%(arm,stage)));assert sha(cp)==meta['sha256']
  raw=torch.load(cp,map_location='cpu');strict[arm][str(stage)]=dict(strict_keys=100,parameters=sum(v.numel() for v in p.parameters()),sha256=sha(cp),steps=meta['steps'])
  if stage==6:
   task6[arm]={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in raw['state_dict'].items()}
   baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict']
   historical_task6[arm]=dict(equal=all(torch.equal(v,baseline[k]) for k,v in raw['state_dict'].items()),max_abs=max(float((v.double()-baseline[k].double()).abs().max()) for k,v in raw['state_dict'].items()),equality_gate=False)
   parity=js(Q/('TASK6_PARITY_'+arm+'.json'));assert parity['same_device_C_equal'] and parity['actual_sha256']==sha(cp)
   continue
  tm=js(Q/('TEACHER_%s_%d.json'%(arm,stage)));tc=torch.load(teacher_path(arm,stage),map_location='cpu');assert tm['source_sha256']==sha(model_path(arm,stage-1))==tc['source_sha256'] and tm['cache_sha256']==sha(teacher_path(arm,stage))
  ids=[i for i,r in enumerate(manifests[stage]) if r['task']<stage and r['bc_valid']];assert ids==tc['ids'] and set(ids)==set(tc['cache']) and len(ids)==OLD[stage]
  targets=torch.stack([torch.as_tensor(stage_labels[stage][i]['next_action']) for i in ids]).float()
  scale=torch.stack([targets[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]]).double();assert torch.allclose(tc['scales'].double(),scale,atol=1e-7,rtol=1e-7)
  ct=np.array([r['teacher_cost'] for r in tm['rows']]);assert abs(tm['tau']-max(1e-8,float(np.median(ct))))<1e-14
  for r in tm['rows']:
   ref=np.float32(tm['tau'])/(np.float32(tm['tau'])+np.float32(r['teacher_cost']))
   assert np.isfinite(r['weight']) and 0<=r['weight']<=1
   assert abs(r['weight']-float(ref))<=float(np.spacing(ref))
  teacher=model(model_path(arm,stage-1));out=[];maxcache=0.
  for start in range(0,len(ids),32):
   chunk=ids[start:start+32];x=torch.stack([torch.load(manifests[stage][i]['path'],map_location='cpu')['feature'].squeeze(0) for i in chunk]);y=torch.stack([torch.as_tensor(stage_labels[stage][i]['next_action']) for i in chunk]).double()
   tq=teacher(x);sq=p(x);calls+=2;windows+=2*len(chunk)
   cq=tuple(torch.stack([tc['cache'][i][j] for i in chunk]) for j in range(3))
   for u,v in zip(tq,cq):
    diff=float((u-v).abs().max());maxcache=max(maxcache,diff);assert torch.allclose(u,v,atol=2e-4,rtol=2e-4),'Cached teacher vs CPU checkpoint mismatch'
   tq=tuple(v.double() for v in tq);sq=tuple(v.double() for v in sq);tmus,tstd,tl=tq;smus,sstd,sl=sq;tk=tl.argmax(-1);sk=sl.argmax(-1);ii=torch.arange(len(chunk))
   def groups(mu,sd):return torch.stack([((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
   tg=groups(tmus,tstd);sg=groups(smus,sstd);anchor=torch.stack([((smus[:,:,a:b]-tmus[ii,tk][:,None,a:b]).square()+(sstd[:,:,a:b]-tstd[ii,tk][:,None,a:b]).square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
   directional,fraction=directional_anchor(smus,sstd,tmus[ii,tk],tstd[ii,tk],y,scale)
   # Independent identity from three squared distances; export scalars only.
   ctotal=sg.sum(-1);dtotal=anchor.sum(-1);tctotal=tg.sum(-1)[ii,tk]
   dot=(dtotal+tctotal[:,None]-ctotal)/2
   alpha=(dot/tctotal[:,None].clamp_min(1e-300)).clamp(0,1)
   gram=dtotal-2*alpha*dot+alpha.square()*tctotal[:,None]
   assert torch.allclose(directional,gram,atol=1e-9,rtol=1e-9)
   assert torch.allclose(fraction,alpha,atol=1e-9,rtol=1e-9)
   for j,i in enumerate(chunk):
    w=tm['rows'][start+j];assert w['index']==i and w['sample_sha256']==manifests[stage][i]['sha256']
    out.append(dict(index=i,task=manifests[stage][i]['task'],sample_sha256=manifests[stage][i]['sha256'],teacher_mode=int(tk[j]),student_mode=int(sk[j]),teacher_component_group_costs=tg[j].tolist(),student_component_group_costs=sg[j].tolist(),student_component_anchor_group_distances=anchor[j].tolist(),student_logits=sl[j].tolist(),teacher_logits=tl[j].tolist(),quality_weight=w['weight'],frozen_teacher_cost=w['teacher_cost'],directional_distances=directional[j].tolist(),projection_fractions=fraction[j].tolist()))
  diags[arm+'_'+str(stage)]=out;teacher_audits[arm+'_'+str(stage)]=dict(count=len(ids),source_sha256=tm['source_sha256'],cache_sha256=tm['cache_sha256'],cache_cpu_max_abs=maxcache,tau=tm['tau'],scale=scale.tolist())
 rec=torch.load(OUTPUT[arm]/'recovery.pth',map_location='cpu');assert rec['stage']==9 and rec['next_epoch']==100 and rec['steps']==105500 and rec['state']['step']==105500 and rec['contract']==sha(Q/'CONTRACT.json') and rec['history']==hist
 assert (OUTPUT[arm]/'recovery.pth').stat().st_ino==model_path(arm,9).stat().st_ino
 assert {'model','optimizer','scheduler','grads','modes','requires','rng','step'}==set(rec['state']) and set(rec['state']['rng'])=={'python','numpy','cpu','cuda'}
 assert all(torch.equal(rec['state_dict'][k],v) for k,v in rec['state']['model'].items())
 norm=lambda names:{n[7:] if n.startswith('module.') else n for n in names}
 assert norm(rec['state']['grads'])==set(dict(p.named_parameters()))==norm(rec['state']['requires'])
 assert rec['teacher_sha256']==sha(teacher_path(arm,9)) and rec['teacher_metadata_sha256']==sha(Q/('TEACHER_%s_9.json'%arm))
 assert rec['state']['optimizer']['state'] and rec['state']['optimizer']['param_groups'] and rec['state']['scheduler']
 assert rec['manifest_sha256']==sha(O/'ra_coreset/dataset_manifests/task9.json')
 norm_modes=lambda names:{'' if n=='module' else n[7:] if n.startswith('module.') else n for n in names}
 assert set(dict(p.named_modules()))<=norm_modes(rec['state']['modes'])
 def finite(x):
  if torch.is_tensor(x):assert torch.isfinite(x).all()
  elif isinstance(x,dict):
   for v in x.values():finite(v)
  elif isinstance(x,(tuple,list)):
   for v in x:finite(v)
 finite(rec['state']['optimizer']);finite(rec['state']['grads'])
 strict[arm]['recovery']=dict(sha256=sha(OUTPUT[arm]/'recovery.pth'),complete_state_fields=sorted(rec['state']),history_exact=True,final_hardlink=True)
assert all(v==task6['C'] for v in task6.values())
records=js(Q/'OUTCOMES.json');assert len(records)==200
cfg=js(O/'resolved_eval.json');sys.path.insert(0,str(S));from libero.libero.benchmark import get_benchmark
with contextlib.redirect_stdout(sys.stderr):b=get_benchmark(cfg['benchmark_name'])(cfg['data']['task_order_index'])
initial={}
for t in range(6,10):
 task=b.get_task(t);fp=Path(cfg['init_states_folder'])/task.problem_folder/task.init_states_file;arr=torch.load(fp,map_location='cpu');initial[t]=(sha(fp),[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in arr[:20]])
seen=set()
for r in records:
 key=(r['arm'],r['stage'],r['task'],r['initial_index']);assert key not in seen;seen.add(key);assert r['attempt']==1 and 0<r['steps']<=600 and (r['success'] or r['steps']==600)
 assert initial[r['task']][0]==r['initial_file_sha256'] and initial[r['task']][1][r['initial_index']]==r['initial_sha256']
 noise=np.random.RandomState(20261024+SEED*1000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32)
 assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256'] and r['cp_sha256']==strict[r['arm']][str(r['stage'])]['sha256']
assert len(seen)==200;verify();light={}
for fp in Q.rglob('*'):
 if fp.is_file() and fp.suffix in ['.json','.jsonl','.exit'] and fp.name not in ['CONTRACT.json','PARENT_BINDING.json'] and fp.stat().st_size<8*1024**2:light[str(fp.relative_to(Q))]=dict(text=fp.read_text(),sha256=sha(fp))
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',strict=strict,contract=sha(Q/'CONTRACT.json'),contract_sha256=sha(Q/'CONTRACT.json'),diagnostics=diags,teacher_audits=teacher_audits,task6_between_current_arms_exact=True,historical_task6=historical_task6,diagnostic_calls=calls,diagnostic_windows=windows,diagnostic_seconds=time.time()-tic,light=light,episodes=200,environment_calls=0,optimizer_updates=0,no_residual=True,processes=[],observed=time.time())))
