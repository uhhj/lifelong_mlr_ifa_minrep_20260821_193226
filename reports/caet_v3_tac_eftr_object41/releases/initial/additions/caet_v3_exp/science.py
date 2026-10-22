"""CAET-v2 gate1. CPU metadata only; no model, optimizer, environment or writes."""
import os,sys,json,time,hashlib,collections,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation as Rot
torch.set_num_threads(1);os.nice(10)

P=Path(os.environ['RA_BASE'])
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
TARGET={6:'butter',7:'milk',8:'chocolate_pudding',9:'orange_juice'}
def layout(xml):
 tree=ET.fromstring(xml);world=tree.find('worldbody');parents={c:p for p in tree.iter() for c in p}
 q=0;result={}
 for n in world.iter():
  if n.tag not in ('joint','freejoint'):continue
  kind='free' if n.tag=='freejoint' else n.get('type','hinge')
  result[n.get('name')]=(q,kind,parents[n]);q+={'free':7,'ball':4,'hinge':1,'slide':1}[kind]
 return world,parents,result
def poses(states,xml,obj):
 world,parents,joints=layout(xml);key=obj+'_1_joint0'
 if key in joints:
  q,kind,body=joints[key];assert kind=='free' and parents[body] is world,(obj,'non-world free joint')
  pos=states[:,1+q:1+q+3];quat=states[:,1+q+3:1+q+7]
  assert np.allclose(np.linalg.norm(quat,axis=1),1,atol=1e-6,rtol=0)
  return pos,Rot.from_quat(quat[:,[1,2,3,0]]),dict(type='free',qpos_offset=q)
 # Basket may be a static world body, but unknown articulated placement is refused.
 bodies=[b for b in world.iter('body') if b.get('name')==obj+'_1'];assert len(bodies)==1,(obj,'unknown body')
 body=bodies[0];assert parents[body] is world and not list(body.findall('joint'))
 assert not any(k in body.attrib for k in ('euler','axisangle','xyaxes','zaxis'))
 pos=np.asarray([float(x) for x in body.get('pos','0 0 0').split()]);quat=np.asarray([float(x) for x in body.get('quat','1 0 0 0').split()])
 return np.repeat(pos[None],len(states),axis=0),Rot.from_quat(np.repeat(quat[[1,2,3,0]][None],len(states),axis=0)),dict(type='static_xml')
def group(e,gap):
 if np.linalg.norm(e[:3])>=.001:axis=int(np.argmax(np.abs(e[:3])));motion='T'+str(axis)+('+' if e[axis]>=0 else '-')
 elif np.linalg.norm(e[3:6])>=np.pi/180:axis=int(np.argmax(np.abs(e[3:6])));motion='R'+str(axis)+('+' if e[3+axis]>=0 else '-')
 else:motion='STATIC'
 progress='+' if e[6]>.001 else '-' if e[6]<-.001 else '0'
 return 'free/'+motion+'/'+progress+'/'+('open' if gap>=.04 else 'closed')
def graph(current,old):
 # Node identities are actual (task,demo,u), never repeated window positions.
 def unique(rows):
  d={}
  for r in sorted(rows,key=lambda x:x['sample']):d.setdefault((r['task'],r['demo'],r['u']),r)
  return list(d.values())
 cur,prev=unique(current),unique(old);edges=[];before=collections.Counter();pruned=collections.Counter();groups={}
 scales=np.array([.01]*3+[np.pi/18]*3+[.01])
 for key in sorted({r['group'] for r in cur}&{r['group'] for r in prev}):
  if '/STATIC/' in key:continue
  for ot in sorted({r['task'] for r in prev}):
   ci=[i for i,r in enumerate(cur) if r['group']==key];oi=[j for j,r in enumerate(prev) if r['group']==key and r['task']==ot]
   if not ci or not oi:continue
   x=np.stack([cur[i]['effect'] for i in ci])/scales;y=np.stack([prev[j]['effect'] for j in oi])/scales
   dist=np.linalg.norm(x[:,None]-y[None,:],axis=2)
   compatible=(dist<=.5)&(np.linalg.norm(np.stack([cur[i]['relative'] for i in ci])[:,None]-np.stack([prev[j]['relative'] for j in oi])[None,:],axis=2)<=.05)&(np.abs(np.array([cur[i]['aperture'] for i in ci])[:,None]-np.array([prev[j]['aperture'] for j in oi])[None,:])<=.01)
   g=str(ot)+'|'+key;before[g]=int(compatible.sum())
   left={i:set(sorted(np.flatnonzero(compatible[i]),key=lambda j:(dist[i,j],prev[oi[j]]['sample']))[:10]) for i in range(len(ci))}
   right={j:set(sorted(np.flatnonzero(compatible[:,j]),key=lambda i:(dist[i,j],cur[ci[i]]['sample']))[:10]) for j in range(len(oi))}
   pairs={(i,j) for i,js_ in left.items() for j in js_ if i in right[j]}
   initial_pairs=sorted(pairs)
   # Iterate support pruning to a fixed point; removal cannot manufacture support.
   while pairs:
    goodc={i for i,j in pairs if len({prev[oi[b]]['demo'] for a,b in pairs if a==i})>=3}
    goodo={j for i,j in pairs if len({cur[ci[a]]['demo'] for a,b in pairs if b==j})>=3}
    new={(i,j) for i,j in pairs if i in goodc and j in goodo}
    if new==pairs:break
    pairs=new
   cids={ci[i] for i,j in pairs};oids={oi[j] for i,j in pairs}
   cd={cur[i]['demo'] for i in cids};od={prev[j]['demo'] for j in oids}
   passed=len(cd)>=3 and len(od)>=3 and len(cids)>=12 and len(oids)>=12
   groups[g]=dict(initial_mutual_edges=[(ci[i],oi[j]) for i,j in initial_pairs], final_pruned_edges=[(ci[i],oi[j]) for i,j in sorted(pairs)],compatible_pairs=before[g],mutual_pruned_pairs=len(pairs),current_demos=sorted(cd),old_demos=sorted(od),current_times=len(cids),old_times=len(oids),qualified=passed)
   if passed:edges.extend((ci[i],oi[j]) for i,j in sorted(pairs))
 supported={i for i,j in edges}
 public=lambda rows:[{k:r[k] for k in ('sample','task','demo','u','group')} for r in rows]
 return dict(current_nodes=public(cur),old_nodes=public(prev),groups=groups,edges=edges,supported_current=len(supported),stage_supported=bool(edges),
  supporting_current_demos=sorted({cur[i]['demo'] for i in supported}),unsupported_current=len(cur)-len(supported))

def derive_task(task):
 b=js(P/'bindings'/f'task{task}.json');hp=b['hdf5'];assert sha(hp)==b['hdf5_sha256']
 demos=sorted({r['demo_id'] for r in b['windows'].values() if r['identity_unique']},key=lambda x:int(x.split('_')[-1]));holdout=demos[-1]
 arrays={};physical={}
 with h5py.File(hp,'r') as h:
  for demo in demos:
   g=h['data'][demo];s=g['states'][:].astype(np.float64);a=g['actions'][:].astype(np.float32)
   j=g['obs/joint_states'][:];grip=g['obs/gripper_states'][:];eef=g['obs/ee_pos'][:]
   assert np.array_equal(s[1:,1:8],j[:-1]) and np.array_equal(s[1:,8:10],grip[:-1])
   xml=g.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml
   pos,rot,pm=poses(s,xml,TARGET[task]);goal,grot,gm=poses(s,xml,'basket')
   # Per-transition effects expressed in the action-before basket frame.
   trans=grot[:-1].inv().apply(np.diff(pos,axis=0))
   worldrot=(rot[1:]*rot[:-1].inv()).as_rotvec();rotation=grot[:-1].inv().apply(worldrot)
   progress=np.linalg.norm(pos[:-1]-goal[:-1],axis=1)-np.linalg.norm(pos[1:]-goal[:-1],axis=1)
   effects=np.concatenate((trans,rotation,progress[:,None]),axis=1)
   arrays[demo]=dict(s=s,a=a,grip=grip,eef=eef,pos=pos,goal=goal,grot=grot,effects=effects)
   physical[demo]=dict(length=len(s),object_layout=pm,goal_layout=gm,post_obs_alignment_exact=True,xml_sha256=hashlib.sha256(xml.encode()).hexdigest())
 rows={};invalid=collections.Counter();proof=[]
 for name,w in b['windows'].items():
  fp=P/'features'/str(task)/name;assert sha(fp)==w['sha256'];v=torch.load(fp,map_location='cpu')
  if not w['identity_unique']:invalid['AMBIGUOUS_IDENTITY']+=1;continue
  ar=arrays[w['demo_id']];u=w['target'];starts=w['possible_starts'];n=len(ar['s'])
  assert all(np.array_equal(v['actions'][0].numpy(),ar['a'][np.minimum(np.arange(st,st+8),n-1)]) for st in starts)
  if any(st<0 or st+7>=n for st in starts):invalid['PADDED_WINDOW']+=1;continue
  if u+1>=len(ar['a']):invalid['NO_NEXT_ACTION']+=1;continue
  assert len(ar['a'])==n
  effect_valid=u+2<n
  assert all(st+7==u for st in starts)
  effect=ar['effects'][u+1] if effect_valid else np.zeros(7);gap=float(abs(ar['grip'][u,0]-ar['grip'][u,1]))
  relative=ar['grot'][u+1].inv().apply(ar['eef'][u]-ar['pos'][u+1])
  assert np.isfinite(effect).all() and np.isfinite(relative).all()
  row=dict(bc_valid=True,valid=effect_valid,full_window_valid=True,identity_unique=True,sample=name,task=task,demo=w['demo_id'],u=u,group=group(effect,gap) if effect_valid else 'INVALID',effect=effect.astype(np.float32),aperture=gap,relative=relative.astype(np.float32),holdout=w['demo_id']==holdout,next_action=ar['a'][u+1].copy(),sample_sha256=w['sha256'])
  rows[name]=row;proof.append(dict(sample=name,demo=w['demo_id'],u=u,next_action_index=u+1,from_state=u+1,to_state=u+2,
     bc_valid=True,effect_valid=effect_valid,full_window_valid=True,possible_starts=starts,max_input_obs_index=u,sample_sha256=w['sha256'],next_action_sha256=hashlib.sha256(ar['a'][u+1].tobytes()).hexdigest(),group=row['group']))
 m=js(P/'ra_coreset/storage'/f'task{task}.json');selected=[]
 for x in m['selected']:
  name=Path(x['source']).name;assert sha(P/'ra_coreset/buffer'/x['destination'])==x['sha256']==b['windows'][name]['sha256']
  if name in rows:selected.append(rows[name])
 return rows, selected, dict(total_windows=len(b['windows']), valid_next_labels=sum(r['valid'] for r in rows.values()),valid_bc_labels=len(rows), invalid=dict(invalid), ra_quota=m['K'],valid_retained=len(selected),holdout_demo=holdout,physical=physical,labels_metadata=proof,binding_sha256=sha(P/'bindings'/f'task{task}.json'),storage_sha256=sha(P/'ra_coreset/storage'/f'task{task}.json'))
