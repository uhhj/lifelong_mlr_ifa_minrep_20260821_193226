"""External CPU-only final audit and fixed expert-cost diagnostic. No server writes."""
import os,sys,ast,json,hashlib,time,contextlib,math
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1/seed42');O=Path('/data/mlr_ifa/experiments/ALIGN_OBJECT_seed42_20261022');S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source')
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
verify();assert (Q.parent/'engineering_copy_20261029_v1/controller.exit').read_text()=='0' and not (Q.parent/'controller.exit').exists() and (Q.parent/'engineering_copy_20261029_v1/HANDOFF_COMPLETE.json').exists()
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

# External audit only. No environment, optimizer, or server write.
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
contract=js(Q/'CONTRACT.json');assert sha(contract['official_initialization']['path'])==contract['official_initialization']['sha256']
assert all((Q/('train_'+a+'.exit')).read_text()=='0' for a in ARMS) and (Q/'evaluate.exit').read_text()=='0'
manifests={};labels={};memory_audit={};strict={};diagnostics={};calls=0;windows=0;tic=time.time();task6={}
for stage in range(6,10):
 mm=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(mm)==POP[stage] and all(sha(r['path'])==r['sha256'] for r in mm)
 ll={}
 for t in range(6,stage):
  pp=O/('effects/memory/task%d.pth'%t);cache=torch.load(pp,map_location='cpu');selected=js(O/('ra_coreset/storage/task%d.json'%t))['selected']
  subset={Path(r['path']).name:r for r in mm if r['task']==t};assert set(subset)==set(cache)=={r['destination'] for r in selected}
  for r in selected:assert r['sha256']==subset[r['destination']]['sha256']==cache[r['destination']]['sample_sha256']
  ll.update({str(O/'ra_coreset/buffer'/n):v for n,v in cache.items()})
 ll.update({str(O/'features'/str(stage)/n):v for n,v in torch.load(O/('effects/current/task%d/labels.pth'%stage),map_location='cpu').items()})
 aligned=[]
 for r in mm:
  lab=ll[r['path']];assert lab['sample_sha256']==r['sha256'] and all(lab[k]==r[k] for k in ['task','demo','u','bc_valid'])
  if lab['bc_valid']:assert lab['identity_unique'] and lab['full_window_valid'] and lab['next_action_index']==lab['u']+1
  aligned.append(lab)
 assert sum(r['bc_valid'] for r in mm)==VALID[stage] and sum(r['bc_valid'] and r['task']<stage for r in mm)==OLD[stage]
 manifests[stage]=mm;labels[stage]=aligned
split=js(Q/'INITIAL_SPLIT.json');actual_initial={}
for t in range(6,10):
 sp=split[str(t)];assert sha(sp['path'])==sp['file_sha256'];arr=torch.load(sp['path'],map_location='cpu');hs={str(i):ah(np.asarray(arr[i])) for i in range(30)}
 assert all(hs[i]==h for i,h in sp['training'].items()) and all(hs[i]==h for i,h in sp['evaluation'].items())
 assert len(set(sp['training'].values()))==10 and not set(sp['training'].values())&set(sp['evaluation'].values());actual_initial[t]=hs
collections=[]
def cost(mu,sd,y,scale):
 return torch.stack([((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
for arm in ARMS:
 strict[arm]={};hist=js(Q/('EPOCHS_'+arm+'.json'));assert len(hist)==400
 completed_steps=0
 for i,h in enumerate(hist):
  task=6+i//100;ep=i%100;batch=(POP[task]+9)//10;completed_steps+=batch
  assert (h['stage'],h['epoch'],h['steps'],h['batches'])==(task,ep,completed_steps,batch)
  order=torch.randperm(POP[task],generator=torch.Generator().manual_seed(ep)).numpy().astype(np.int64)
  assert ah(order)==h['sampler_sha256']
  assert h['totals']['valid']==VALID[task] and h['totals']['invalid']==POP[task]-VALID[task] and h['totals']['old_eligible']==OLD[task]
 cached={}
 for stage in range(6,10):
  cp=model_path(arm,stage);policy=model(cp);cp_sha=sha(cp);meta=js(Q/('STAGE_%s_%d.json'%(arm,stage)));assert meta['sha256']==cp_sha
  strict[arm][str(stage)]=dict(strict_keys=100,parameters=sum(v.numel() for v in policy.parameters()),sha256=cp_sha,steps=meta['steps'])
  if stage==6:task6[arm]={k:ah(v.numpy()) for k,v in policy.state_dict().items()}
  # Immutable acquisition cache: actual CPU replay under its source checkpoint.
  key='%s_%d'%(arm,stage);mp=memory_path(arm,stage);cache=torch.load(mp,map_location='cpu');m=js(Q/('MEMORY_'+key+'.json'))
  assert sha(mp)==m['cache_sha256'] and cache['source_sha256']==m['source_sha256']==cp_sha and cache['contract']==sha(Q/'CONTRACT.json')
  assert len(cache['rows'])==len(m['rows'])==m['count']<=128 and cache['task']==stage and cache['arm']==arm
  current=[i for i,r in enumerate(manifests[stage]) if r['task']==stage and r['bc_valid']]
  yy=torch.stack([torch.as_tensor(labels[stage][i]['next_action']) for i in current]).float()
  scale=torch.stack([yy[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]])
  assert torch.equal(scale,cache['scales']) and np.allclose(scale.numpy(),m['scales'],atol=1e-8,rtol=1e-8)
  rr=js(Q/('COLLECTION_OUTCOMES_'+key+'.json'));assert len(rr)==10 and sha(Q/('COLLECTION_OUTCOMES_'+key+'.json'))==m['collection_outcomes_sha256']
  assert (Q/('collect_'+key+'.exit')).read_text()=='0' and js(Q/('COLLECTION_STATE_PARITY_'+key+'.json'))['parent_all_state_exact']
  chosen={};quotas=[13]*8+[12]*2
  for j,r in enumerate(rr):
   assert (r['arm'],r['task'],r['initial_index'],r['attempt'],r['env_seed'])==(arm,stage,20+j,1,SEED+100)
   assert r['initial_file_sha256']==split[str(stage)]['file_sha256'] and r['initial_sha256']==actual_initial[stage][str(20+j)] and r['cp_sha256']==cp_sha
   assert 0<r['steps']<=600 and r['termination']==('success' if r['success'] else 'timeout600') and (r['success'] or r['steps']==600)
   noise=np.random.RandomState(20271025+SEED*1000+stage*100+20+j).standard_normal((600,7)).astype(np.float32);assert ah(noise)==r['noise_sha256']
   gen=np.random.RandomState((90100+SEED)+stage*100+20+j);pool=[]
   for step in range(r['steps']):
    n=step+1;slot=step if n<=quotas[j] else int(gen.randint(n))
    if slot<quotas[j]:
     if n<=quotas[j]:pool.append(step)
     else:pool[slot]=step
   chosen[j]=set(pool) if arm=='O' or r['success'] else set()
  collections.extend(rr);expected_demo=set()
  if arm=='D':
   demos=sorted({labels[stage][i]['demo'] for i in current})
   for j,demo in enumerate(demos):
    ids=sorted([i for i in current if labels[stage][i]['demo']==demo],key=lambda i:labels[stage][i]['u']);quota=128//len(demos)+int(j<128%len(demos))
    expected_demo.update(ids[int(pos)] for pos in np.random.RandomState((90100+SEED)+stage*100+j).choice(len(ids),size=min(quota,len(ids)),replace=False))
   assert {r['source_index'] for r in cache['rows']}==expected_demo
  else:assert {(r['trajectory'],r['step']) for r in cache['rows']}=={(j,s) for j,steps in chosen.items() for s in steps}
  maxdiff=0.
  for row,public in zip(cache['rows'],m['rows']):
   assert all(row[k]==v for k,v in public.items() if k not in ['feature_sha256','mu_sha256','sd_sha256'])
   for name in ['feature','mu','sd']:assert ah(row[name].numpy())==public[name+'_sha256']
   assert tuple(row['feature'].shape)==(row['length'],4,512) and 1<=row['length']<=8 and row['mu'].shape==row['sd'].shape==(7,)
   if arm=='D':
    idx=row['source_index'];native=torch.load(manifests[stage][idx]['path'],map_location='cpu')['feature'].squeeze(0).float();assert torch.equal(native,row['feature']) and row['source_sha256']==manifests[stage][idx]['sha256']
   else:assert row['length']==min(row['step']+1,8) and row['initial_index']==20+row['trajectory']
   mu,sd,z=policy(row['feature'][None]);calls+=1;windows+=1;k=int(z[0].argmax());assert k==row['mode']
   for x,y in [(mu[0,k],row['mu']),(sd[0,k],row['sd'])]:
    maxdiff=max(maxdiff,float((x-y).abs().max()));assert torch.allclose(x,y,atol=2e-4,rtol=2e-4)
  cached[stage]=cache;memory_audit[key]=dict(count=m['count'],cache_sha256=sha(mp),source_sha256=cp_sha,CPU_teacher_max_abs=maxdiff,sampling_exact=True,scale_exact=True,successes=sum(r['success'] for r in rr),no_raw_image_replay=True)
  if stage>6:
   inp=js(Q/('TRAIN_INPUT_'+key+'.json'));ids=[i for i,r in enumerate(manifests[stage]) if r['bc_valid'] and r['task']<stage]
   yy=torch.stack([torch.as_tensor(labels[stage][i]['next_action']) for i in ids]).float();sc=torch.stack([yy[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]]).double()
   assert np.allclose(sc.numpy(),inp['scale'],atol=1e-7,rtol=1e-7)
   out=[]
   for start in range(0,len(ids),32):
    ii=ids[start:start+32];xx=torch.stack([torch.load(manifests[stage][i]['path'],map_location='cpu')['feature'].squeeze(0) for i in ii]);yy=torch.stack([torch.as_tensor(labels[stage][i]['next_action']) for i in ii]).double()
    mu,sd,z=(x.double() for x in policy(xx));calls+=1;windows+=len(ii);cc=cost(mu,sd,yy,sc)
    for j,i in enumerate(ii):out.append(dict(index=i,task=manifests[stage][i]['task'],sample_sha256=manifests[stage][i]['sha256'],student_mode=int(z[j].argmax()),component_group_costs=cc[j].tolist(),logits=z[j].tolist()))
   diagnostics['expert_'+key]=out;out=[]
   assert set(inp['memory'])=={str(t) for t in range(6,stage)}
   for task in range(6,stage):
    prior=cached[task];assert inp['memory'][str(task)]['cache_sha256']==sha(memory_path(arm,task)) and inp['memory'][str(task)]['source_sha256']==sha(model_path(arm,task))
    for i,row in enumerate(prior['rows']):
     mu,sd,z=(x.double() for x in policy(row['feature'][None]));calls+=1;windows+=1;tm=row['mu'].double();ts=row['sd'].double();sc=prior['scales'].double()
     dd=torch.stack([((mu[:,:,a:b]-tm[a:b]).square()+(sd[:,:,a:b]-ts[a:b]).square()).mean(-1)/sc[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
     out.append(dict(task=task,index=i,feature_sha256=ah(row['feature'].numpy()),student_mode=int(z[0].argmax()),component_group_distances=dd[0].tolist(),logits=z[0].tolist()))
   diagnostics['retain_'+key]=out
 assert sum(len(c['rows']) for c in cached.values())<=512

 # Independent gate, support, fixed-probe, and physical-cost audit.
 gates=[json.loads(line) for line in (Q/'GATES.jsonl').read_text().splitlines()]
 assert len(gates)==9100
 physical=0;offset=0;by_stage={}
 for stage in range(6,10):
  nb=(POP[stage]+9)//10
  if stage>6:
   pr=js(Q/('PROBES_%d.json'%stage));ids=[i for i,r in enumerate(manifests[stage]) if r['task']==stage and r['bc_valid']]
   gen=np.random.RandomState(20261027+SEED*100+stage)
   assert pr['new_indices']==sorted(int(x) for x in gen.choice(ids,min(64,len(ids)),replace=False))
   yy=torch.stack([torch.as_tensor(labels[stage][i]['next_action']).float() for i in ids])
   sc=torch.stack([yy[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]])
   assert np.allclose(sc.numpy(),pr['new_scale'],atol=1e-7,rtol=1e-7) and pr['new_population']==len(ids)
   assert pr['source']=='TRAINING_ONLY_FIXED_NOT_VALIDATION'
   assert set(pr['old_indices'])==set(pr['support'])==set(pr['support_trajectories'])=={str(t) for t in range(6,stage)}
   for t in range(6,stage):
    rows=cached[t]['rows'];meta=js(Q/('MEMORY_S_%d.json'%t));trajectories=sorted({int(x['trajectory']) for x in rows})
    assert meta['training_attempts']==10 and len(trajectories)==meta['collection_successes']
    assert pr['support'][str(t)]==len(trajectories)/10. and pr['support_trajectories'][str(t)]==trajectories
    gen=np.random.RandomState(20261027+SEED*1000+t)
    expected=sorted(int(x) for x in gen.choice(len(rows),min(8,len(rows)),replace=False)) if rows else []
    assert pr['old_indices'][str(t)]==expected
    binding=js(Q/('TRAIN_INPUT_S_%d.json'%stage))['memory'][str(t)]
    assert binding['support']==len(trajectories)/10. and binding['support_trajectories']==trajectories
  subset=[g for g in gates if g['stage']==stage];accepted_q=0
  for ep in range(100):
   rr=hist[(stage-6)*100+ep];gg=[g for g in subset if g['epoch']==ep]
   assert [g['batch'] for g in gg]==(list(range(0,nb,10)) if stage>6 else [])
   kept=0
   for g in gg:
    step=offset+ep*nb+g['batch'];assert g['start_step']==step and g['steps']==min(10,nb-g['batch']) and g['end_step']==step+g['steps']
    c,q=g['C'],g['Q'];checks=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
    assert all(math.isfinite(x[k]) for x in [c,q] for k in ['bc_new','exec_new','retain_support'])
    assert checks==g['checks'] and g['accepted']==('Q' if all(checks.values()) else 'C')
    if g['accepted']=='Q':kept+=g['steps'];accepted_q+=1
   physical+=nb*(2 if stage>6 else 1)
   assert rr['physical_optimizer_calls']==physical and rr['totals']['anchor_slots']==kept*4
  offset+=100*nb
  if stage>6:by_stage[str(stage)]=dict(blocks=len(subset),Q_accepts=accepted_q,support=pr['support'],fixed_probes_verified=True)
 assert physical==195300 and offset==105500
 gate_audit=dict(status='PASS_INDEPENDENT_RECOMPUTATION',blocks=9100,physical_training_optimizer_calls=physical,accepted_updates=offset,by_stage=by_stage)
 rec=torch.load(OUTPUT[arm]/'recovery.pth',map_location='cpu');assert rec['stage']==9 and rec['next_epoch']==100 and rec['steps']==rec['state']['step']==rec['retention_sampling_cursor']==105500 and rec['history']==hist and rec['contract']==sha(Q/'CONTRACT.json')
 assert set(rec['state'])=={'model','optimizer','scheduler','grads','modes','requires','rng','step'} and set(rec['state']['rng'])=={'python','numpy','cpu','cuda'}
 assert (OUTPUT[arm]/'recovery.pth').stat().st_ino==model_path(arm,9).stat().st_ino
 assert all(torch.equal(rec['state_dict'][k],x) for k,x in rec['state']['model'].items())
 norm=lambda names:{'' if n=='module' else n[7:] if n.startswith('module.') else n for n in names}
 assert norm(rec['state']['grads'])==set(dict(policy.named_parameters()))==norm(rec['state']['requires'])
 assert set(dict(policy.named_modules()))<=norm(rec['state']['modes']) and all(isinstance(x,bool) for x in rec['state']['modes'].values())
 assert rec['state']['optimizer']['state'] and rec['state']['optimizer']['param_groups'] and rec['state']['scheduler']
 def finite_state(x):
  if torch.is_tensor(x):assert torch.isfinite(x).all()
  elif isinstance(x,dict):
   for y in x.values():finite_state(y)
  elif isinstance(x,(list,tuple)):
   for y in x:finite_state(y)
 finite_state(rec['state']['optimizer']);finite_state(rec['state']['grads'])
 assert rec['physical_optimizer_calls']==195300 and rec['gate_blocks']==9100 and rec['probe_sha256']==sha(Q/'PROBES_9.json')
 assert rec['memory_bindings']==js(Q/('TRAIN_INPUT_%s_9.json'%arm))['memory']
 strict[arm]['recovery']=dict(sha256=sha(OUTPUT[arm]/'recovery.pth'),complete_state_fields=sorted(rec['state']),history_exact=True,final_hardlink=True,sampling_cursor=rec['retention_sampling_cursor'])
for arm in ARMS:
 path_ad=js(Q/('SHORT_HISTORY_ADMISSION_'+arm+'.json'));assert path_ad['status']=='PASS' and path_ad['eight_frame_legacy_exact'] and path_ad['all_state_restored'] and [x['length'] for x in path_ad['lengths']]==list(range(1,9))
assert len(task6)==1 and len(collections)==40
records=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()
for r in records:
 key=(r['arm'],r['stage'],r['task'],r['initial_index']);assert key not in seen;seen.add(key)
 assert r['arm'] in ARMS and 6<=r['task']<=r['stage']<=9 and r['initial_index'] in range(20) and r['attempt']==1
 assert 0<r['steps']<=600 and r['termination']==('success' if r['success'] else 'timeout600') and (r['success'] or r['steps']==600)
 assert r['env_seed']==SEED+r['initial_index']//10 and r['cp_sha256']==strict[r['arm']][str(r['stage'])]['sha256']
 assert r['initial_file_sha256']==split[str(r['task'])]['file_sha256'] and r['initial_sha256']==actual_initial[r['task']][str(r['initial_index'])]
 noise=np.random.RandomState(20261024+SEED*1000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32);assert ah(noise)==r['noise_sha256']

verify();light={}
for fp in Q.rglob('*'):
 if fp.is_file() and fp.suffix in ['.json','.jsonl','.exit'] and fp.name not in ['CONTRACT.json','PARENT_BINDING.json'] and fp.stat().st_size<8*1024**2:light[str(fp.relative_to(Q))]=dict(text=fp.read_text(),sha256=sha(fp))
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',strict=strict,gate_audit=gate_audit,memory_audits=memory_audit,contract_sha256=sha(Q/'CONTRACT.json'),diagnostics=diagnostics,diagnostic_calls=calls,diagnostic_windows=windows,diagnostic_seconds=time.time()-tic,task6_current_arms_exact=False,single_S_only=True,initial_state_bytes_verified=True,reference_C_pairing_exact=False,light=light,episodes=200,training_collection_episodes=40,environment_calls=0,optimizer_updates=0,no_residual=True,processes=[],observed=time.time(),limitation='Online retained-feature hashes and teacher CPU replay verified; no raw camera or full physics replay is claimed.')))
