"""Current-task-only deterministic evidence, never differences of augmented caches."""
import argparse,time
import h5py
from .common import *
CAMERAS=('agentview_rgb','eye_in_hand_rgb')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int,choices=range(6,10));a=ap.parse_args();t=a.task;verify();capacity();seed()
 out=ROOT/'evidence'/('task%d'%t);assert not out.exists(),'Evidence cannot be silently overwritten';out.mkdir(parents=True)
 binding=js(PARENT/'bindings'/('task%d.json'%t));assert sha(binding['hdf5'])==binding['hdf5_sha256']
 demos=sorted({w['demo_id'] for w in binding['windows'].values() if w.get('identity_unique',True)});assert len(demos)==10
 import robomimic.utils.obs_utils as ObsUtils
 from libero.lifelong.metric import raw_obs_to_tensor_obs
 from libero.lifelong.utils import get_task_embs
 c=config();ObsUtils.initialize_obs_utils_with_obs_specs({'obs':c.data.obs.modality});front=full_frontend();arrays={};count=0;tic=time.time()
 with h5py.File(binding['hdf5'],'r') as h:
  lang=json.loads(h['data'].attrs['problem_info'])['language_instruction'];tok=get_task_embs(c,[lang]);instr=tok['input_ids'][0];mask=tok['attention_mask'][0]
  for demo in demos:
   g=h['data'][demo];acts=g['actions'][:].astype(np.float32);r_native=np.concatenate([g['obs/joint_states'][:],g['obs/gripper_states'][:]],-1)
   states=g['states'][:];assert np.array_equal(states[1:,1:10],r_native[:-1]);r=r_native.astype(np.float32);n=len(acts);assert len(r)==n;vv=[]
   for start in range(0,n,32):
    obs=[{c.data.obs_key_mapping[k]:g['obs'][k][i] for k in c.data.obs_key_mapping} for i in range(start,min(n,start+32))]
    data=raw_obs_to_tensor_obs(obs,instr,mask,c);data['obs']={k:v.unsqueeze(1) for k,v in data['obs'].items()};data=front.preprocess_input(data,train_mode=False)
    with torch.no_grad():
     feat=torch.stack([front.image_proj(front.image_encoders(data['obs'][cam][:,0],langs=None)) for cam in CAMERAS],1)
     if start==0:
      check=torch.stack([front.image_proj(front.image_encoders(data['obs'][cam][:,0],langs=None)) for cam in CAMERAS],1);assert torch.equal(feat,check)
    assert feat.shape==(len(obs),2,512);vv.append(feat.cpu());count+=2*len(obs)
   arrays[demo]=dict(action=torch.from_numpy(acts),response=torch.from_numpy(r),phi=torch.cat(vv));event('evidence_demo',task=t,demo=demo,frames=n)
 del front;torch.cuda.empty_cache()
 if t==6:
  # PCA sees only the first eight identity-sorted Task6 demos, never validation or later tasks.
  proj=[]
  for camera in range(2):
   z=torch.cat([arrays[d]['phi'][:,camera] for d in demos[:8]]).double();z=z-z.mean(0);_,_,v=torch.linalg.svd(z,full_matrices=False);v=v[:32].T
   signs=torch.sign(v.gather(0,v.abs().argmax(0)[None]).squeeze(0));proj.append((v*signs).float())
  projections=torch.stack(proj);save(ROOT/'shared/projection.pth',projections)
 else:
  assert sha(ROOT/'shared/projection.pth')==js(ROOT/'CODEC_FREEZE_MANIFEST.json')['projection_sha256'];projections=torch.load(ROOT/'shared/projection.pth',map_location='cpu')
 rows={};proof=[];first_valid=0;mask_counts={k:0 for k in ('action','response','visual')}
 for name,w in binding['windows'].items():
  path=PARENT/'features'/str(t)/name;assert sha(path)==w['sha256'];payload=torch.load(path,map_location='cpu');assert payload['feature'].shape==(1,8,4,512)
  unique=w.get('identity_unique',True);starts=w['possible_starts'];u=w['target'];demo=w['demo_id'];arr=arrays.get(demo);valid=bool(unique and starts and all(s>=0 and s+7<w['length'] and s+7==u for s in starts))
  row=dict(task=t,sample=name,demo=demo,u=u,sample_sha256=w['sha256'],source_hdf5_sha256=binding['hdf5_sha256'])
  for k,dim in [('action',7),('response',9),('visual',64)]:row[k]=torch.zeros(4,dim);row[k+'_mask']=torch.zeros(4,dtype=torch.bool)
  if unique:
   assert arr is not None and len(arr['action'])==w['length']
   assert all(torch.equal(payload['actions'][0].cpu(),arr['action'][torch.arange(s,s+8).clamp(max=w['length']-1)]) for s in starts)
  for k in range(4):
   j=u+1+k
   if valid and j<len(arr['action']):row['action'][k]=arr['action'][j];row['action_mask'][k]=True
   if valid and j<len(arr['response']):row['response'][k]=arr['response'][j]-arr['response'][j-1];row['response_mask'][k]=True
   if valid and j<len(arr['phi']):
    delta=arr['phi'][j]-arr['phi'][j-1];row['visual'][k]=torch.einsum('cd,cde->ce',delta,projections).flatten();row['visual_mask'][k]=True
  assert bool(row['action_mask'][0])==bool(valid and u+1<w['length']);first_valid+=int(row['action_mask'][0])
  for k in mask_counts:mask_counts[k]+=int(row[k+'_mask'].sum())
  # Retrieval is observation-only, pre-FiLM visual evidence plus measured robot state at u.
  row['support']=torch.cat([arr['phi'][max(0,min(u,len(arr['phi'])-1))].flatten(),arr['response'][max(0,min(u,len(arr['response'])-1))]]) if unique else torch.zeros(1033)
  row['identity_valid']=valid;rows[name]=row
  proof.append(dict(sample=name,demo=demo,u=u,source_sha256=w['sha256'],identity_valid=valid,indices=[u+1+k for k in range(4)],masks={k:row[k+'_mask'].tolist() for k in mask_counts},first_action_sha256=tensor_sha(row['action'][0]) if row['action_mask'][0] else None))
 save(out/'labels.pth',rows)
 if t==6:
  stats={}
  for k in ('action','response','visual'):
   values=torch.cat([r[k][r[k+'_mask']] for r in rows.values() if r['demo'] in demos[:8]])
   stats[k+'_mean']=values.mean(0);stats[k+'_std']=values.std(0,unbiased=False).clamp_min(1e-4)
  save(ROOT/'shared/normalization.pth',stats)
 report=dict(status='PASS',task=t,current_demos=demos,codec_train_demos=demos[:8] if t==6 else None,codec_validation_demos=demos[8:] if t==6 else None,windows=len(rows),first_step_valid=first_valid,mask_counts=mask_counts,labels_sha256=sha(out/'labels.pth'),binding_sha256=sha(PARENT/'bindings'/('task%d.json'%t)),hdf5_sha256=binding['hdf5_sha256'],official_sha256=sha(c.pretrain),projection_sha256=sha(ROOT/'shared/projection.pth'),normalization_sha256=sha(ROOT/'shared/normalization.pth'),pre_FiLM=True,deterministic_repeat_exact=True,unique_camera_forwards=count,repeat_camera_forwards=2*sum(min(32,len(arrays[d]['action'])) for d in demos),environment_steps=0,optimizer_steps=0,seconds=time.time()-tic,proof=proof)
 write(out/'TEMPORAL_BINDING_AUDIT.json',report);event('evidence_complete',task=t,windows=len(rows),seconds=report['seconds'])
if __name__=='__main__':main()
