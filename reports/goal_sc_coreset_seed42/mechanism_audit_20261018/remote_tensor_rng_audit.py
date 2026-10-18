"""Read existing trusted checkpoint/RNG files on CPU; no model forward or environment."""
import json,hashlib,collections,time
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(2)
SC=Path('/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018')
P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017')
OUT=Path('/tmp/sc_mechanism_tensor_rng_20261018.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
files=[]
for root,arm in [(SC,'sc_coreset'),(P,'coreset')]:
 files+=list((root/arm/'models').glob('*ep99.pth'))+list((root/arm/'evaluation').glob('rng_before*.pth'))
files+=list((SC/'bindings').glob('*.json'))
before={str(p):sha(p) for p in files};start=time.time()
def digest(x):
 h=hashlib.sha256()
 def walk(v):
  if torch.is_tensor(v):
   v=v.cpu().contiguous();h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,(tuple,list)):
   h.update(type(v).__name__.encode())
   for z in v:walk(z)
  else:h.update(repr(v).encode())
 walk(x);return h.hexdigest()
rng=[]
for s in range(6,10):
 for t in range(6,s+1):
  filename=f'rng_before_stage{s}_task{t}.pth'
  x=torch.load(P/'coreset/evaluation'/filename,map_location='cpu');y=torch.load(SC/'sc_coreset/evaluation'/filename,map_location='cpu')
  d={k:{'coreset':digest(x[k]),'sc':digest(y[k]),'equal':digest(x[k])==digest(y[k])} for k in x}
  rng.append({'stage':s,'task':t,'streams':d})
def category(k):
 if 'temporal_transformer' in k:return 'temporal_transformer'
 for tag in ['mean_layer','logstd_layer','logits_layer','share']:
  if tag in k:return 'head_'+tag
 return 'other'
models=[]
for s in range(6,10):
 x=torch.load(P/'coreset/models'/f'mlrifa_{s}_model_ep99.pth',map_location='cpu')['state_dict']
 y=torch.load(SC/'sc_coreset/models'/f'mlrifa_{s}_model_ep99.pth',map_location='cpu')['state_dict']
 assert x.keys()==y.keys();stats={};neq=0
 for k in x:
  assert x[k].shape==y[k].shape
  neq+=not torch.equal(x[k],y[k])
  if not x[k].is_floating_point() or k.endswith(('attn.bias','masked_bias')):continue
  u=x[k].double();v=y[k].double();diff=v-u;c=category(k)
  row=stats.setdefault(c,dict(n=0,squared_delta=0.,squared_parent=0.,max_abs=0.,keys=0))
  row['n']+=u.numel();row['keys']+=1;row['squared_delta']+=float(diff.square().sum());row['squared_parent']+=float(u.square().sum());row['max_abs']=max(row['max_abs'],float(diff.abs().max()))
 for v in stats.values():
  v['relative_l2']=(v['squared_delta']/max(v['squared_parent'],1e-30))**.5
  v['rms_delta']=(v['squared_delta']/v['n'])**.5
 models.append({'stage':s,'different_tensor_keys':neq,'total_tensor_keys':len(x),'groups':stats})
 del x,y
actions={}
for t in range(6,10):
 b=json.loads((SC/'bindings'/f'task{t}.json').read_text())['windows']
 m=json.loads((SC/'sc_coreset/storage'/f'task{t}.json').read_text())['selection']
 groups={'full':list(b),'coreset':m['original_ids'],'sc':m['selected_ids'],'removed':sorted(set(m['original_ids'])-set(m['selected_ids'])),'added':sorted(set(m['selected_ids'])-set(m['original_ids']))}
 report={}
 for name,ids in groups.items():
  aa=np.array([b[i]['last_action'] for i in ids],dtype=np.float64)
  # Aggregate descriptors only. Never export the per-window expert action vectors.
  report[name]={'n':len(ids),'mean':aa.mean(0).tolist(),'std':aa.std(0).tolist(),'translation_norm_mean':float(np.linalg.norm(aa[:,:3],axis=1).mean()),'rotation_norm_mean':float(np.linalg.norm(aa[:,3:6],axis=1).mean()),'near_zero_translation_fraction':float((np.linalg.norm(aa[:,:3],axis=1)<.01).mean())}
 actions[str(t)]=report
after={str(p):sha(p) for p in files};assert before==after
result={'status':'PASS','cpu_only':True,'model_forwards':0,'optimizer_updates':0,'environment_calls':0,'files_unchanged':True,'input_sha256':before,'rng':rng,'models':models,'action_aggregates':actions,'seconds':time.time()-start,'time':time.time(),'limits':'Tensor distances are descriptive, not behavioral evidence. RNG equality covers snapshots before a cell, not per-episode or per-step isolation.'}
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','output':str(OUT),'files':len(files),'seconds':result['seconds']}))
