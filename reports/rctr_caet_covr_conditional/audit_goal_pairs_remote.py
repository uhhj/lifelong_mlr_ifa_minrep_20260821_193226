"""CPU-only, read-only replay of frozen CAET GOAL label and batch support.

Sources are supplied through stdin by the fixed-scope client. No model is built,
no optimizer is constructed, no simulator is imported, and no file is written.
"""
import ast,collections,glob,hashlib,json,math,os,random,sys,time
from pathlib import Path
from typing import List,Sequence
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
torch.set_num_threads(1)
os.nice(10)
REQUEST=json.loads(sys.stdin.read())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def extract(source,names):
 tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-pure-functions>','exec'),globals())
extract(REQUEST['labels'],['joint_layout','recorded_effects','certified'])
extract(REQUEST['transfer'],['pair_masks','transfer_loss'])
DIMENSIONS={'hinge':(1,1),'slide':(1,1),'ball':(4,3),'free':(7,6)}
TARGETS={6:'cream_cheese',7:'flat_stove_button',8:'akita_black_bowl',9:'wine_bottle'}
class FileTrajectoryDataset:
 def __init__(self,file_paths):self.file_paths=file_paths
extract(REQUEST['native'],['build_dataset_for_task'])
def _list_pt(p):return list(VIRTUAL_FILES[p])
def public(row):return {k:row[k] for k in ('task','demo','target','sample','moving','holdout')}
def support(rows,edges,current):
 participating=set(i for edge in edges for i in edge)
 cross=[(i,j) for i,j in edges if rows[i]['task']==current or rows[j]['task']==current]
 def coverage(indices):
  return dict(windows=len(indices),unique_targets=len({(rows[i]['task'],rows[i]['demo'],rows[i]['target']) for i in indices}),
   demos=sorted({str(rows[i]['task'])+':'+rows[i]['demo'] for i in indices}))
 groups={};demo_pairs=collections.Counter()
 for i,j in edges:
  x,y=sorted((rows[i],rows[j]),key=lambda r:r['task']);key='%d-%d'%(x['task'],y['task']);groups[key]=groups.get(key,0)+1
  demo_pairs['%d:%s|%d:%s'%(x['task'],x['demo'],y['task'],y['demo'])]+=1
 cp={i for e in cross for i in e if rows[i]['task']==current}
 op={i for e in cross for i in e if rows[i]['task']!=current}
 return dict(unordered_pairs=len(edges),by_task_pair=groups,current_old_pairs=len(cross),old_old_pairs=len(edges)-len(cross),
  current_support=coverage(cp),old_support=coverage(op),all_support=coverage(participating),demo_pair_counts=dict(demo_pairs),
  current_moving_windows=sum(x['task']==current and x['moving'] for x in rows),
  current_train_demos=sorted({x['demo'] for x in rows if x['task']==current}),
  current_target_pair_count=len({(rows[i]['demo'],rows[i]['target'],rows[j]['task'],rows[j]['demo'],rows[j]['target']) for i,j in cross}))
report=dict(kind='ZERO_TRAINING_FROZEN_BATCH_REPLAY',started_unix=time.time(),optimizer_calls=0,environment_steps=0,
 server_writes=0,device='CPU',cuda_initialized=False,sources={k:hashlib.sha256(REQUEST[k].encode()).hexdigest() for k in ('labels','transfer','native','fit')},seeds={})
for seed in (41,42,43):
 root=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_%s'%(seed,'20261018' if seed==42 else '20261019'))
 labels={};current_paths={};memory_paths={};bindings={};proof={};protected={}
 for task in range(6,10):
  bp=root/'bindings'/('task%d.json'%task);b=js(bp);protected[str(bp)]=digest(bp)
  assert digest(b['hdf5'])==b['hdf5_sha256'];protected[b['hdf5']]=b['hdf5_sha256']
  demos=sorted({x['demo_id'] for x in b['windows'].values()},key=lambda s:int(s.split('_')[-1]));holdout=demos[-1]
  values={};actions={};aligned={}
  with h5py.File(b['hdf5'],'r') as f:
   for demo in demos:
    values[demo],aligned[demo]=recorded_effects(f['data'][demo],TARGETS[task]);actions[demo]=f['data'][demo]['actions'][:].astype(np.float32)
  rows={}
  for name,x in b['windows'].items():
   assert certified(x)
   fp=root/'features'/str(task)/name;assert digest(fp)==x['sha256']
   data=torch.load(fp,map_location='cpu')['actions'][0].numpy();raw=actions[x['demo_id']]
   assert data.shape==(8,7) and len(raw)==x['length']
   assert all(np.array_equal(data,raw[np.minimum(np.arange(start,start+8),len(raw)-1)]) for start in x['possible_starts'])
   target=x['target'];valid=target<len(values[x['demo_id']]);effect=values[x['demo_id']][target].astype(np.float32) if valid else None
   rows[name]=dict(task=task,demo=x['demo_id'],target=target,sample=name,valid=valid,holdout=x['demo_id']==holdout,
    moving=bool(valid and (np.linalg.norm(effect[:3])>=.001 or np.linalg.norm(effect[3:])>=np.pi/180)),effect=effect)
  labels[task]=rows
  current_paths[task]={str(root/'features'/str(task)/name):x for name,x in rows.items()}
  manifest=root/'ra_coreset/storage'/('task%d.json'%task);m=js(manifest);protected[str(manifest)]=digest(manifest)
  memory_paths[task]={}
  for x in m['selected']:
   name=Path(x['source']).name;p=root/'ra_coreset/buffer'/x['destination']
   assert digest(p)==x['sha256']==b['windows'][name]['sha256']
   memory_paths[task][str(p)]=rows[name]
  proof[str(task)]=dict(windows=len(rows),valid=sum(x['valid'] for x in rows.values()),moving=sum(x['moving'] for x in rows.values()),
   holdout_demo=holdout,binding_sha256=protected[str(bp)],storage_sha256=protected[str(manifest)],cached_window_and_buffer_identity='PASS',recorded_physical_alignment='PASS')
 stages={}
 for stage in (7,8,9):
  mapping=dict(current_paths[stage]);buffer={k:v for task in range(6,stage) for k,v in memory_paths[task].items()};mapping.update(buffer)
  base=str(root/'features');buf=str(root/'ra_coreset/buffer')
  VIRTUAL_FILES={os.path.join(base,str(stage)):sorted(current_paths[stage]),buf:sorted(buffer)}
  native=build_dataset_for_task(base,stage,buf,seed,start=6).file_paths
  expected=sorted(buffer)+sorted(current_paths[stage]);random.Random(seed).shuffle(expected);assert native==expected
  payload=[mapping[p] for p in native if mapping[p]['valid']]
  effects=torch.tensor(np.stack([x['effect'] for x in payload]));moving=torch.tensor([x['moving'] for x in payload]);tasks=torch.tensor([x['task'] for x in payload])
  train_ids=torch.tensor([i for i,x in enumerate(payload) if not x['holdout']]);val_ids=torch.tensor([i for i,x in enumerate(payload) if x['holdout']])
  center=effects[train_ids].double().mean(0).float();scale=effects[train_ids].double().std(0,unbiased=False).float().clamp_min(1e-6);norm=(effects-center)/scale
  tr=[payload[i] for i in train_ids];pos,neg=pair_masks(norm[train_ids],moving[train_ids],tasks[train_ids])
  edges=[tuple(x) for x in torch.nonzero(torch.triu(pos,diagonal=1)).tolist()]
  distance=torch.cdist(norm[train_ids],norm[train_ids]);eligible=(tasks[train_ids,None]!=tasks[train_ids][None,:]) & moving[train_ids,None] & moving[train_ids][None,:]
  margin=float((distance[eligible]-.5).abs().min()) if eligible.any() else None
  epochs=[];edge_seen=collections.Counter();current_anchors=set();old_anchors=set()
  for epoch in range(10):
   order=train_ids[torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(820000000+10000*seed+100*stage+epoch))]
   batches=[]
   for start in range(0,len(order),64):
    ids=order[start:start+64];pp,nn=pair_masks(norm[ids],moving[ids],tasks[ids]);eligible_anchor=pp.any(-1)&nn.any(-1)
    actual=pp&eligible_anchor[:,None];pairs=torch.nonzero(actual).tolist()
    cur_old=sum(int(tasks[ids[i]])==stage or int(tasks[ids[j]])==stage for i,j in pairs)
    cur_eligible=[]
    for i,j in pairs:
     left,right=int(ids[i]),int(ids[j]);edge_seen[tuple(sorted((left,right)))]+=1
     if int(tasks[left])==stage:current_anchors.add(left);cur_eligible.append(left)
     else:old_anchors.add(left)
    # Same original transfer_loss eligibility/count with inert embeddings, no gradients/model.
    inert=torch.zeros(len(ids),1);_,used=transfer_loss(inert,norm[ids],moving[ids],tasks[ids]);assert used==len(pairs)
    batches.append(dict(index=start//64,size=len(ids),payload_indices=ids.tolist(),negative_supported_payload_indices=ids[nn.any(-1)].tolist(),positive_directed=int(pp.sum()),used_directed=len(pairs),current_old_used_directed=cur_old,
     old_old_used_directed=len(pairs)-cur_old,current_eligible_anchors=len(set(cur_eligible)),
     used_pairs=[[int(ids[i]),int(ids[j])] for i,j in pairs]))
   epochs.append(dict(epoch=epoch,order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),batches=batches))
  # Independent held-out current demo support against old TRAIN labels; never fit on holdout.
  held=[]
  for i in val_ids.tolist():
   if payload[i]['task']!=stage or not payload[i]['moving']:continue
   dd=torch.cdist(norm[i:i+1],norm[train_ids])[0];mask=(tasks[train_ids]!=stage)&moving[train_ids]&(dd<=.5)
   if mask.any():held.append(dict(sample=payload[i]['sample'],demo=payload[i]['demo'],target=payload[i]['target'],old_training_matches=int(mask.sum())))
  sums=lambda key:sum(b[key] for ep in epochs for b in ep['batches'])
  stage_report=dict(training_valid=len(train_ids),heldout_valid=len(val_ids),global_positive_directed=int(pos.sum()),global_negative_directed=int(neg.sum()),
   support=support(tr,edges,stage),positive_edges_train_indices=edges,train_metadata=[public(x) for x in tr],
   payload_metadata=[public(x) for x in payload],train_payload_indices=train_ids.tolist(),
   minimum_positive_threshold_margin=margin,native_order_sha256=hashlib.sha256('\n'.join(native).encode()).hexdigest(),
   normalization_scale=scale.tolist(),epochs=epochs,batch_summary=dict(total_batches=sum(len(x['batches']) for x in epochs),
    used_directed=sums('used_directed'),current_old_used_directed=sums('current_old_used_directed'),old_old_used_directed=sums('old_old_used_directed'),
    batches_with_any=sum(b['used_directed']>0 for ep in epochs for b in ep['batches']),
    batches_with_current_old=sum(b['current_old_used_directed']>0 for ep in epochs for b in ep['batches']),
    unique_current_anchors=len(current_anchors),current_anchor_demos=sorted({payload[i]['demo'] for i in current_anchors}),
    unique_seen_pairs=len(edge_seen)),heldout_current_support=held)
  assert stage_report['global_positive_directed']==REQUEST['previous']['seeds'][str(seed)]['stages'][str(stage)]['positive_pairs']
  stages[str(stage)]=stage_report
 report['seeds'][str(seed)]=dict(tasks=proof,stages=stages,protected_metadata_hashes=protected)
 print('Completed seed '+str(seed)+' CPU label/batch audit',file=sys.stderr,flush=True)
assert not torch.cuda.is_initialized()
report['finished_unix']=time.time();report['status']='PASS';print(json.dumps(report,allow_nan=False))
