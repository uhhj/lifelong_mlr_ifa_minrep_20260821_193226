import argparse,hashlib,collections
import numpy as np
import torch,h5py
from .common import *
from .state_labels import recorded
from .effect_labels import effects,SCHEMA
from .split_by_demo import splits

def state_modules():
 from torch import nn
 enc=nn.Sequential(nn.Linear(9,512),nn.ReLU(inplace=True),nn.Linear(512,512))
 scale=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512));shift=__import__('copy').deepcopy(scale)
 raw=torch.load(config().pretrain,map_location='cpu')['state_dict']
 for prefix,m in [('state_encoder.',enc),('state_scale.',scale),('state_shift.',shift)]:
  m.load_state_dict({k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)},strict=True);m.eval().requires_grad_(False)
 return enc,scale,shift

def resolve_window(w,payload,observations,actions,modules):
 """Disambiguate repeated terminal actions using frozen proprio token, never labels."""
 candidates=[];enc,scale,shift=modules;lang=payload['feature'][0,:,0];want=payload['feature'][0,:,3]
 ds=sorted({x[0] for x in w['candidate_identities']}) if not w['identity_unique'] else [w['demo_id']]
 for demo in ds:
  a=actions[demo];n=len(a)
  starts=range(n) if not w['identity_unique'] else w['possible_starts']
  for st in starts:
   idx=np.minimum(np.arange(st,st+8),n-1)
   if not np.array_equal(payload['actions'][0].numpy(),a[idx]):continue
   obs=observations[demo];proprio=torch.tensor(np.concatenate([obs['joint_states'][idx],obs['gripper_states'][idx]],-1),dtype=torch.float32)
   with torch.no_grad():reconstructed=(1+scale(lang))*enc(proprio)+shift(lang)
   delta=float((reconstructed-want).abs().max());match=torch.allclose(reconstructed,want,atol=1e-4,rtol=1e-5)
   candidates.append(dict(demo=demo,start=int(st),delta=delta,match=bool(match)))
 matches=[r for r in candidates if r['match']];assert matches,('no proprio match',w['sample_id'],candidates)
 chosen=matches[0];base=observations[chosen['demo']];idx=np.minimum(np.arange(chosen['start'],chosen['start']+8),len(actions[chosen['demo']])-1)
 for r in matches[1:]:
  other=observations[r['demo']];jj=np.minimum(np.arange(r['start'],r['start']+8),len(actions[r['demo']])-1)
  assert r['demo']==chosen['demo'] and all(np.array_equal(base[k][idx],other[k][jj]) for k in base),('indistinguishable proprio but different visual identity',w['sample_id'])
 return chosen,candidates

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--suite',default='OBJECT',choices=['OBJECT']);ap.add_argument('--seed',type=int,default=41,choices=[41]);ap.parse_args()
 contract=verify();result={};allsplits={};capacity(600*1024**2);modules=state_modules()
 for task in (6,7):
  b=js(PARENT/'bindings'/('task%d.json'%task));hp=b['hdf5'];assert sha(hp)==b['hdf5_sha256'];arrays={};physical={};observations={}
  demos=sorted({w['demo_id'] for w in b['windows'].values() if w['identity_unique']},key=lambda s:int(s.split('_')[-1]));allsplits[str(task)]=splits(demos)
  with h5py.File(hp,'r') as h:
   for demo in demos:
    g=h['data'][demo];q,obj,eef,gap,xml=recorded(g,task);a=g['actions'][:].astype(np.float32);e=effects(obj,eef,gap)
    assert len(a)==len(q)
    arrays[demo]=(q,obj,e,a)
    observations[demo]={k:g['obs'][k][:] for k in ('joint_states','gripper_states','agentview_rgb','eye_in_hand_rgb')}
    physical[demo]=dict(xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),frames=len(q),post_action_alignment_exact=True)
  names=sorted(b['windows']);meta=[];qrows=[];prevrows=[];erows=[];arows=[];dynamic=[];mask=[]
  for name in names:
   w=dict(b['windows'][name]);fp=PARENT/'features'/str(task)/name;assert sha(fp)==w['sha256'];payload=torch.load(fp,map_location='cpu');resolution=None
   if not w['identity_unique'] or len(w['possible_starts'])!=1:
    chosen,proof=resolve_window(w,payload,observations,{d:v[3] for d,v in arrays.items()},modules)
    w.update(demo_id=chosen['demo'],possible_starts=[chosen['start']],target=min(chosen['start']+7,len(arrays[chosen['demo']][0])-1));resolution=proof
   q,obj,e,a=arrays[w['demo_id']];u=w['target'];n=len(q);starts=w['possible_starts'];assert starts
   indices=[np.minimum(np.arange(st,st+8),n-1) for st in starts];assert all(np.array_equal(i,indices[0]) for i in indices),'ambiguous padded window cannot be re-exported'
   assert payload['feature'].shape==(1,8,4,512)
   assert all(np.array_equal(payload['actions'][0].numpy(),a[idx]) for idx in indices)
   full=all(st>=0 and st+7<n and st+7==u for st in starts)
   sv=bool(full and u+1<n);av=bool(full and u+1<len(a));ev=bool(sv and av and u+2<n)
   row=dict(task_id=task,demo_id=w['demo_id'],sample=name,window_start=starts[0],last_observation_index=u,current_state_index=u+1,target_action_index=u+1,next_state_index=u+2,
    source_feature_sha256=w['sha256'],source_hdf5_sha256=b['hdf5_sha256'],state_valid=sv,action_valid=av,effect_valid=ev,full_window=full,proprio_identity_resolution=resolution)
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
