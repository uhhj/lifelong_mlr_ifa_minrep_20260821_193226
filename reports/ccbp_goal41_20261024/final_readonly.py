"""External CPU-only final audit and fixed expert-cost diagnostic. No server writes."""
import os,sys,ast,json,hashlib,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1');P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1');O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022');S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
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
verify();assert (Q/'controller.exit').read_text()=='0'
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  assert int(p.name)==os.getpid() or str(Q) not in cmd,('Residual',p.name)
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
manifest=js(O/'ra_coreset/dataset_manifests/task9.json')['rows'];assert len(manifest)==3365
assert all(sha(r['path'])==r['sha256'] for r in manifest)
labels=torch.load(O/'effects/memory/task8.pth',map_location='cpu');storage=js(O/'ra_coreset/storage/task8.json')['selected'];md={Path(r['path']).name:r for r in manifest if r['task']==8}
assert set(labels)==set(md)=={r['destination'] for r in storage} and len(labels)==452
rows=[]
for m in storage:
 r=labels[m['destination']];d=md[m['destination']];assert m['sha256']==d['sha256']==r['sample_sha256'] and all(d[k]==r[k] for k in ['task','demo','u','bc_valid'])
 if r['bc_valid']:
  assert r['identity_unique'] and r['full_window_valid'] and r['next_action_index']==r['u']+1
  rows.append((m,r))
cache=torch.load(P/'teacher_cache.pth',map_location='cpu');assert sha(P/'teacher_cache.pth')==js(Q/'TEACHER.json')['cache_sha256']
scale=cache['scales'].double();targets=torch.stack([r['next_action'] for m,r in rows]).float();expected=torch.stack([targets[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]]).double();assert torch.allclose(scale,expected,atol=1e-7,rtol=1e-7)
teacher=model(O/'ra_coreset/models/mlrifa_8_model_ep99.pth');diags={};strict={};tic=time.time();calls=0
for arm in 'ABCD':
 p=model(P/arm/'stage9.pth');train=js(Q/('TRAIN_'+arm+'.json'));assert sha(P/arm/'stage9.pth')==train['cp_sha256'];rec=torch.load(P/arm/'recovery.pth',map_location='cpu')
 assert rec['next_epoch']==100 and rec['state']['step']==33700 and rec['contract']==sha(Q/'CONTRACT.json') and rec['teacher_sha256']==sha(P/'teacher_cache.pth')
 raw=torch.load(P/arm/'stage9.pth',map_location='cpu')['state_dict'];assert all(torch.equal(raw[k],v) for k,v in rec['state']['model'].items())
 assert set(rec['state']['grads'])==set(raw)==set(rec['state']['requires']) and set(rec['state']['rng'])=={'python','numpy','cpu','cuda'}
 strict[arm]=dict(strict_keys=100,cp_sha256=train['cp_sha256'],recovery_sha256=sha(P/arm/'recovery.pth'),complete_state_fields=sorted(rec['state']),parameters=sum(v.numel() for v in p.parameters()))
 out=[]
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32];x=torch.stack([torch.load(O/'ra_coreset/buffer'/m['destination'],map_location='cpu')['feature'].squeeze(0) for m,r in chunk]);y=torch.stack([r['next_action'] for m,r in chunk]).double()
  qq=[tuple(v.double() for v in f(x)) for f in [teacher,p]];calls+=2
  parts=[]
  for mu,sd,l in qq:
   k=l.argmax(-1);ix=torch.arange(len(k));group=torch.stack([((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
   mix=-torch.logsumexp(l.log_softmax(-1)+D.Normal(mu,sd).log_prob(y[:,None]).sum(-1),-1)
   parts.append((k,group,mix))
  for j,(m,r) in enumerate(chunk):
   tk,tg,tn=parts[0];sk,sg,sn=parts[1]
   out.append(dict(sample=m['destination'],sample_sha256=m['sha256'],teacher_mode=int(tk[j]),student_mode=int(sk[j]),teacher_component_group_costs=tg[j].tolist(),student_component_group_costs=sg[j].tolist(),teacher_mixture_nll=float(tn[j]),student_mixture_nll=float(sn[j])))
 diags[arm]=out
records=js(Q/'OUTCOMES.json');assert len(records)==160
cfg=js(O/'resolved_eval.json');sys.path.insert(0,str(S));from libero.libero.benchmark import get_benchmark
b=get_benchmark(cfg['benchmark_name'])(cfg['data']['task_order_index']);seen=set()
for r in records:
 key=(r['arm'],r['task'],r['initial_index']);assert key not in seen;seen.add(key);assert r['attempt']==1 and 0<r['steps']<=600 and (r['success'] or r['steps']==600)
 task=b.get_task(r['task']);fp=Path(cfg['init_states_folder'])/task.problem_folder/task.init_states_file;arr=torch.load(fp,map_location='cpu')[r['initial_index']]
 assert sha(fp)==r['initial_file_sha256'] and hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==r['initial_sha256']
 noise=np.random.RandomState(20261024+41000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32)
 assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256'] and r['cp_sha256']==strict[r['arm']]['cp_sha256']
assert len(seen)==160;verify();light={}
for fp in Q.rglob('*'):
 if fp.is_file() and fp.suffix in ['.json','.jsonl','.exit'] and fp.name!='CONTRACT.json' and fp.stat().st_size<8*1024**2:light[str(fp.relative_to(Q))]=dict(text=fp.read_text(),sha256=sha(fp))
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',strict=strict,contract=sha(Q/'CONTRACT.json'),diagnostics=diags,diagnostic_calls=calls,diagnostic_windows=len(rows)*8,diagnostic_seconds=time.time()-tic,scale=scale.tolist(),manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json'),dataset_counts={str(t):dict(population=sum(r['task']==t for r in manifest),valid=sum(r['task']==t and r['bc_valid'] for r in manifest)) for t in [6,7,8,9]},light=light,episodes=160,environment_calls=0,optimizer_updates=0,no_residual=True,observed=time.time())))
