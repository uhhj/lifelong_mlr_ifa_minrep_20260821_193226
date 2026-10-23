"""Exact formal preprocessing, no augmentation in eval, frozen official frontend."""
import argparse,copy,hashlib,time
import h5py
from .common import *

def native_batch(policy,obs,instr,mask):
 from libero.lifelong.metric import raw_obs_to_tensor_obs
 c=policy.cfg;data=raw_obs_to_tensor_obs(obs,instr,mask,c)
 data['obs']={k:v.unsqueeze(1) for k,v in data['obs'].items()}
 return policy.spatial_encode(policy.preprocess_input(data,train_mode=False))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',type=int,default=[6,7]);ap.add_argument('--seed',type=int,default=41);a=ap.parse_args();assert a.tasks==[6,7] and a.seed==41
 verify();capacity(600*1024**2);seed(41)
 import robomimic.utils.obs_utils as ObsUtils
 from libero.lifelong.utils import get_task_embs
 ObsUtils.initialize_obs_utils_with_obs_specs({'obs':config().data.obs.modality})
 c=config();p=full_policy();assert not p.training and not any(m.training for m in p.modules())
 rows=[];begin=time.time();framecount=0
 for task in (6,7):
  labels=torch.load(OUT/'private'/('labels%d.pt'%task));b=js(PARENT/'bindings'/('task%d.json'%task));demos=sorted({r['demo_id'] for r in labels['metadata']});encoded={}
  with h5py.File(b['hdf5'],'r') as h:
   lang=json.loads(h['data'].attrs['problem_info'])['language_instruction'];tok=get_task_embs(c,[lang]);instr=tok['input_ids'][0];mask=tok['attention_mask'][0]
   for demo in demos:
    g=h['data'][demo];n=len(g['actions']);frames=[]
    for start in range(0,n,32):
     obs=[{c.data.obs_key_mapping[k]:g['obs'][k][i] for k in c.data.obs_key_mapping} for i in range(start,min(n,start+32))]
     with torch.no_grad():
      y=native_batch(p,obs,instr,mask)
      if start==0:
       y2=native_batch(p,obs,instr,mask);assert torch.equal(y,y2),'eval preprocessing must be deterministic'
     assert y.shape==(len(obs),1,4,512);frames.append(y[:,0].cpu());framecount+=len(obs)
    encoded[demo]=torch.cat(frames)
    log('native_demo_export',task=task,demo=demo,frames=n)
  windows=[]
  for r in labels['metadata']:
   f=encoded[r['demo_id']];idx=torch.arange(r['window_start'],r['window_start']+8).clamp(max=len(f)-1);assert int(idx.min())>=0
   windows.append(f[idx])
  x=torch.stack(windows);assert x.shape==(len(labels['metadata']),8,4,512)
  file=OUT/'private'/('native%d.pt'%task);save(file,x)
  rows.append(dict(task=task,windows=len(x),frames=sum(len(v) for v in encoded.values()),sha256=sha(file),per_window_sha256=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in x],original_binding_sha256=sha(PARENT/'bindings'/('task%d.json'%task)),hdf5_sha256=b['hdf5_sha256']))
 write(OUT/'NATIVE_VIEW_MANIFEST.json',dict(status='PASS',tasks=rows,official_sha256=sha(c.pretrain),strict_full_official_keys=len(p.state_dict()),frontend_eval=True,exact_formal_preprocessing=True,repeat_exact=True,unique_observation_forwards=framecount,elapsed_seconds=time.time()-begin,optimizer_calls=0,environment_steps=0))
 log('native_complete',frames=framecount)
if __name__=='__main__':main()
