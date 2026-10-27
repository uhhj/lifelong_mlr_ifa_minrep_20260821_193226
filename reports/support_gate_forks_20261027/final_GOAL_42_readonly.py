import os,sys,ast,json,hashlib,time,contextlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1/GOAL_42');O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed42_20261022');S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_42/source')
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
# Appended to independent CPU-native policy loader; source contract checked outside.
import math
tic=time.time();calls=windows=0
binding=js(Q/'BINDING.json');stage=STAGE
assert all(sha(p)==h for p,h in binding['files'].items())
common=torch.load(DATA_PRIVATE/CASE/'COMMON_INITIAL.pth',map_location='cpu')
origin=torch.load(binding['checkpoint'],map_location='cpu')['state_dict'];assert all(torch.equal(x,common['model'][k]) for k,x in origin.items())
assert common['step']==binding['checkpoint_steps'] and common['optimizer']['state']=={}
memory={};cachechecks={}
for task in range(6,stage):
    b=binding['caches'][str(task)];m=torch.load(b['path'],map_location='cpu');teacher=model(b['teacher']);meta=b['metadata'];support=len({r['trajectory'] for r in m['rows']})/10.
    assert support==meta['collection_successes']/10. and len(m['rows'])==meta['count']
    maxdiff=0.
    for row in m['rows']:
        mu,sd,z=teacher(row['feature'][None]);k=int(z[0].argmax());calls+=1;windows+=1
        diff=max(float((mu[0,k]-row['mu']).abs().max()),float((sd[0,k]-row['sd']).abs().max()));maxdiff=max(maxdiff,diff)
        assert torch.allclose(mu[0,k],row['mu'],atol=2e-4,rtol=2e-4) and torch.allclose(sd[0,k],row['sd'],atol=2e-4,rtol=2e-4)
    cachechecks[str(task)]=dict(count=len(m['rows']),support=support,teacher_sha256=sha(b['teacher']),cache_sha256=sha(b['path']),max_teacher_cpu_abs=maxdiff);m['support']=support;memory[task]=m
manifest=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(manifest)==POP[stage]
labels={}
for task in range(6,stage):
    fp=O/('effects/memory/task%d.pth'%task);stored=torch.load(fp,map_location='cpu');selected=js(O/('ra_coreset/storage/task%d.json'%task))['selected']
    assert set(stored)=={r['destination'] for r in selected}
    labels.update({str(O/'ra_coreset/buffer'/n):r for n,r in stored.items()})
labels.update({str(O/'features'/str(stage)/n):r for n,r in torch.load(O/('effects/current/task%d/labels.pth'%stage),map_location='cpu').items()})
rows=[labels[r['path']] for r in manifest]
for m,r in zip(manifest,rows):
    assert sha(m['path'])==m['sha256']==r['sample_sha256'] and all(m[k]==r[k] for k in ['task','demo','u','bc_valid'])
assert sum(r['bc_valid'] for r in rows)==VALID[stage]
assert sum(r['bc_valid'] and r['task']<stage for r in rows)==OLD[stage]
probes=js(Q/'PROBES.json');cur=[i for i,r in enumerate(rows) if r['task']==stage and r['bc_valid']]
gen=np.random.RandomState(20261027+SEED*100+stage);assert sorted(int(i) for i in gen.choice(cur,min(64,len(cur)),replace=False))==probes['new_indices']
assert probes['support']=={str(t):m['support'] for t,m in memory.items()}
strict={};diagnostics={};models={}
def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
for arm in ARMS:
    assert (Q/('train_'+arm+'.exit')).read_text()=='0'
    hist=js(Q/('EPOCHS_'+arm+'.json'));assert len(hist)==100
    rec=torch.load(OUTPUT[arm]/'recovery.pth',map_location='cpu');assert rec['history']==hist and rec['next_epoch']==100 and rec['contract']==sha(Q/'CONTRACT.json')
    expected=binding['checkpoint_steps']+100*((POP[stage]+9)//10);assert rec['state']['step']==expected
    assert rec['physical_optimizer_calls']==100*((POP[stage]+9)//10)*(2 if arm=='G' else 1)
    assert set(rec['state'])=={'model','optimizer','scheduler','grads','modes','requires','rng','step'}
    assert rec['state']['optimizer']['state'] and rec['state']['rng'] and rec['state']['scheduler'];finite(rec['state']['model']);finite(rec['state']['optimizer']);finite(rec['state']['grads'])
    p=model(model_path(arm,stage));models[arm]={k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in p.state_dict().items()}
    assert (OUTPUT[arm]/'recovery.pth').stat().st_ino==model_path(arm,stage).stat().st_ino
    for ep,h in enumerate(hist):
        order=torch.randperm(POP[stage],generator=torch.Generator().manual_seed(ep)).numpy().astype(np.int64)
        assert hashlib.sha256(order.tobytes()).hexdigest()==h['sampler_sha256']
    initial=js(Q/('INITIAL_'+arm+'.json'));assert initial['common_state_sha256']==sha(DATA_PRIVATE/CASE/'COMMON_INITIAL.pth')
    strict[arm]=dict(strict_keys=100,parameters=sum(x.numel() for x in p.parameters()),sha256=sha(model_path(arm,stage)),complete_recovery=True,common_initial_exact=True,accepted_updates=expected-binding['checkpoint_steps'],physical_optimizer_calls=rec['physical_optimizer_calls'])
    # Fixed valid expert windows from both old and new task; scalar-only output.
    diag=[]
    for task in range(6,stage+1):
        ids=[i for i,r in enumerate(rows) if r['task']==task and r['bc_valid']]
        yall=torch.stack([torch.as_tensor(rows[i]['next_action']).double() for i in ids]);scale=torch.stack([yall[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]])
        for start in range(0,len(ids),32):
            ii=ids[start:start+32];x=torch.stack([torch.load(manifest[i]['path'],map_location='cpu')['feature'].squeeze(0) for i in ii]);y=torch.stack([torch.as_tensor(rows[i]['next_action']).double() for i in ii]);mu,sd,z=p(x);calls+=1;windows+=len(ii);mu=mu.double();sd=sd.double();k=z.argmax(-1)
            costs=sum(((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)]))
            for j,i in enumerate(ii):diag.append(dict(index=i,task=task,sample_sha256=manifest[i]['sha256'],component_costs=costs[j].tolist(),logits=z[j].tolist(),selected=int(k[j]),executed_cost=float(costs[j,k[j]]),regret=float(costs[j,k[j]]-costs[j].min())))
    diagnostics[arm]=diag
if all(m['support']==1. for m in memory.values()):assert models['S']==models['Q'],'q=1 complete run must match S'
assert (Q/'evaluate.exit').read_text()=='0';records=js(Q/'OUTCOMES.json');assert len(records)==len(ARMS)*(stage-5)*20
cfg=js(O/'resolved_eval.json');sys.path.insert(0,str(S));from libero.libero.benchmark import get_benchmark
with contextlib.redirect_stdout(sys.stderr):bench=get_benchmark(cfg['benchmark_name'])(cfg['data']['task_order_index'])
initials={}
for task in range(6,stage+1):
    t=bench.get_task(task);fp=Path(cfg['init_states_folder'])/t.problem_folder/t.init_states_file;arr=torch.load(fp,map_location='cpu')
    initials[task]=(sha(fp),[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arr[:20]])
seen=set()
for r in records:
    key=(r['arm'],r['task'],r['initial_index']);assert key not in seen;seen.add(key)
    assert r['stage']==stage and r['attempt']==1 and r['cp_sha256']==strict[r['arm']]['sha256']
    assert r['initial_file_sha256']==initials[r['task']][0] and r['initial_sha256']==initials[r['task']][1][r['initial_index']]
    eps=np.random.RandomState(20261024+SEED*1000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32)
    assert hashlib.sha256(eps.tobytes()).hexdigest()==r['noise_sha256']
light={}
for fp in Q.rglob('*'):
    if fp.is_file() and fp.suffix in ['.json','.jsonl','.exit'] and fp.name not in ['BINDING.json','CONTRACT.json']:
        assert fp.stat().st_size<12*1024**2;light[str(fp.relative_to(Q))]=dict(text=fp.read_text(),sha256=sha(fp))
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL',strict=strict,cachechecks=cachechecks,contract=sha(Q/'CONTRACT.json'),diagnostics=diagnostics,light=light,episodes=len(records),diagnostic_calls=calls,diagnostic_windows=windows,diagnostic_seconds=time.time()-tic,new_optimizer_updates=0,new_environments=0)))
