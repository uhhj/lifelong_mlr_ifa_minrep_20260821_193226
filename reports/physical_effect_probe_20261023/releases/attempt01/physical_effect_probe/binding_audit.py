import argparse,hashlib,collections
import numpy as np
import torch,h5py
from .common import *
from .state_labels import recorded
from .effect_labels import effects,SCHEMA
from .split_by_demo import splits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--suite',default='OBJECT',choices=['OBJECT']);ap.add_argument('--seed',type=int,default=41,choices=[41]);ap.parse_args()
 contract=verify();result={};allsplits={};capacity(600*1024**2)
 for task in (6,7):
  b=js(PARENT/'bindings'/('task%d.json'%task));hp=b['hdf5'];assert sha(hp)==b['hdf5_sha256'];arrays={};physical={}
  demos=sorted({w['demo_id'] for w in b['windows'].values()},key=lambda s:int(s.split('_')[-1]));allsplits[str(task)]=splits(demos)
  with h5py.File(hp,'r') as h:
   for demo in demos:
    g=h['data'][demo];q,obj,eef,gap,xml=recorded(g,task);a=g['actions'][:].astype(np.float32);e=effects(obj,eef,gap)
    assert len(a)==len(q)
    arrays[demo]=(q,obj,e,a)
    physical[demo]=dict(xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),frames=len(q),post_action_alignment_exact=True)
  names=sorted(b['windows']);meta=[];qrows=[];prevrows=[];erows=[];arows=[];dynamic=[];mask=[]
  for name in names:
   w=b['windows'][name];assert w['identity_unique'],('ambiguous identity',task,name)
   q,obj,e,a=arrays[w['demo_id']];u=w['target'];n=len(q);starts=w['possible_starts'];assert starts
   indices=[np.minimum(np.arange(st,st+8),n-1) for st in starts];assert all(np.array_equal(i,indices[0]) for i in indices),'ambiguous padded window cannot be re-exported'
   fp=PARENT/'features'/str(task)/name;assert sha(fp)==w['sha256'];payload=torch.load(fp,map_location='cpu')
   assert payload['feature'].shape==(1,8,4,512)
   assert all(np.array_equal(payload['actions'][0].numpy(),a[idx]) for idx in indices)
   full=all(st>=0 and st+7<n and st+7==u for st in starts)
   sv=bool(full and u+1<n);av=bool(full and u+1<len(a));ev=bool(sv and av and u+2<n)
   row=dict(task_id=task,demo_id=w['demo_id'],sample=name,window_start=starts[0],last_observation_index=u,current_state_index=u+1,target_action_index=u+1,next_state_index=u+2,
    source_feature_sha256=w['sha256'],source_hdf5_sha256=b['hdf5_sha256'],state_valid=sv,action_valid=av,effect_valid=ev,full_window=full)
   meta.append(row);qrows.append(q[u] if sv else np.zeros(7));prevrows.append(q[u-1] if sv and u>0 else np.zeros(7));erows.append(e[u] if ev else np.zeros(7));arows.append(a[u+1] if av else np.zeros(7))
   dynamic.append(bool(sv and u>0 and np.linalg.norm(obj[u+1]-obj[u])>=.001));mask.append((sv,av,ev))
  storage=js(PARENT/'ra_coreset/storage'/('task%d.json'%task));retained=[]
  for r in storage['selected']:
   name=Path(r['source']).name;assert r['sha256']==b['windows'][name]['sha256'];assert sha(PARENT/'ra_coreset/buffer'/r['destination'])==r['sha256'];retained.append(names.index(name))
  assert len(names)=={6:1562,7:1480}[task] and len(retained)=={6:781,7:740}[task]
  tensors={k:torch.tensor(np.asarray(v),dtype=torch.float32) for k,v in [('q',qrows),('previous_q',prevrows),('effect',erows),('bc_target',arows)]}
  tensors.update(mask=torch.tensor(mask,dtype=torch.bool),dynamic=torch.tensor(dynamic),metadata=meta,retained_indices=retained,schema=SCHEMA)
  assert all(torch.isfinite(tensors[k]).all() for k in ('q','previous_q','effect','bc_target'))
  p=OUT/'private'/('labels%d.pt'%task);save(p,tensors)
  write(OUT/('BINDING_TASK%d.json'%task),meta)
  result[str(task)]=dict(population=len(names),state_valid=sum(x[0] for x in mask),action_valid=sum(x[1] for x in mask),effect_valid=sum(x[2] for x in mask),dynamic=sum(dynamic),ordered_ra_members=[names[i] for i in retained],physical=physical,labels_sha256=sha(p),binding_sha256=sha(PARENT/'bindings'/('task%d.json'%task)),hdf5_sha256=b['hdf5_sha256'])
 write(OUT/'DEMO_SPLITS.json',allsplits);write(OUT/'INPUT_BINDING_AUDIT.json',dict(status='PASS',tasks=result,contract_sha256=contract,schema=SCHEMA,environment_steps=0,optimizer_calls=0))
 log('binding_complete',counts={k:{a:b for a,b in v.items() if isinstance(b,int)} for k,v in result.items()})
if __name__=='__main__':main()
