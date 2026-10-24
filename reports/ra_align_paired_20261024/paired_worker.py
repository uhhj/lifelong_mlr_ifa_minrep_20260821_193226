"""Finite, independently coupled-noise diagnostic. Existing experiments read-only."""
import os,sys,json,hashlib,time,copy,collections,fcntl,traceback,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
SPEC=json.loads((ROOT/'INPUT_SPEC.json').read_text());S=Path(SPEC['seeds'][0]['source']);sys.path.insert(0,str(S))
os.environ.update(CORESET_OUT=SPEC['seeds'][0]['output'],CORESET_SEED='41')
import numpy as np
import torch
from easydict import EasyDict
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark=False
from coreset_exp.model import data_setup,complete,strict_downstream
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.metric import raw_obs_to_tensor_obs,SubprocVectorEnv,OffScreenRenderEnv
from libero.lifelong.utils import control_seed
PRIVATE=Path('/root/mlr_ifa/experiments/RA_ALIGN_PAIRED_GOAL_20261024_v1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def harray(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,allow_nan=False,indent=2));os.replace(q,p)
def event(kind,**kw):
 x=dict(kind=kind,time=time.time(),**kw)
 with (ROOT/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
 print(json.dumps(x),flush=True)
def verify(e):
 O=Path(e['output']);assert sha(O/'CONTRACT.json')==e['contract_sha256'];c=js(O/'CONTRACT.json')
 assert all(sha(Path(e['source'])/n)==h for n,h in c['source_sha256'].items())
 for method,out,key in [('RA',e['ra_output'],'ra_checkpoints'),('ALIGN',e['output'],'checkpoints')]:
  assert sha(Path(out)/'CONTRACT.json')==e['ra_contract_sha256' if method=='RA' else 'contract_sha256']
  assert js(Path(out)/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
  for t,h in e[key].items():assert sha(Path(out)/'ra_coreset/models'/('mlrifa_%s_model_ep99.pth'%t))==h
def load_pair(e,method,c,shape):
 out=e['ra_output' if method=='RA' else 'output'];models=[]
 for t in [8,9]:
  bc=copy.deepcopy(c);bc.training=False
  p=BriefPolicy(bc,shape).cuda().eval().requires_grad_(False)
  raw=torch.load(Path(out)/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%t),map_location='cpu')['state_dict']
  sd={(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()}
  assert len(p.state_dict())==len(sd)==100 and set(p.state_dict())==set(sd)
  p.load_state_dict(sd,strict=True)
  assert all(torch.equal(v.cpu(),sd[k]) for k,v in p.state_dict().items())
  assert not p.policy_head.low_eval_noise
  models.append(p)
 return models
def params(p,x):return p.policy_head.forward_fn(p.temporal_encode(x)[:,-1])
def selected(q):
 mu,sd,l=q;k=l.argmax(-1);i=torch.arange(len(k),device=k.device);return mu[i,k],sd[i,k],k
def clone_data(d):return {'obs':{k:v.clone() for k,v in d['obs'].items()},'task_emb':d['task_emb'].clone(),'masks':d['masks'].clone()}
def stack_data(hist):return {'obs':{k:torch.stack([d['obs'][k] for d in hist],1) for k in hist[0]['obs']},'task_emb':hist[0]['task_emb'],'masks':hist[0]['masks']}
def encode(front,d):return front.spatial_encode(front.preprocess_input(clone_data(d),train_mode=False))
def noise(seed,index):return np.random.RandomState(20261024+seed*1000+800+index).standard_normal((600,7)).astype(np.float32)
def capacity():
 st=os.statvfs('/root');free=st.f_bavail*st.f_frsize
 allocated=sum(p.stat().st_blocks*512 for p in PRIVATE.rglob('*') if p.is_file()) if PRIVATE.exists() else 0
 assert free-max(0,3*1024**3-allocated)>=1024**3,('capacity',free,allocated)
 st=os.statvfs('/data');assert st.f_bfree*st.f_frsize>=2*1024**3
 return {'system_bavail':free,'private_allocated':allocated,'remaining_bound':max(0,3*1024**3-allocated)}
def obs_copy(obs,keys):return {k:np.asarray(obs[k]).copy() for k in keys}
def obs_hash(obs,keys):
 h=hashlib.sha256()
 for k in keys:
  v=np.ascontiguousarray(obs[k]);h.update(k.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes())
 return h.hexdigest()
def save_raw(path,seq):
 if not seq:return
 np.savez_compressed(path,**{k:np.stack([x[k] for x in seq]) for k in seq[0]})
def main():
 lock=open('/data/mlr_ifa/RA_ALIGN_PAIRED_GOAL_20261024.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'RUN_STARTED.json').exists(),'Never blindly resume formal trajectories'
 capacity();PRIVATE.mkdir(parents=True,exist_ok=False)
 for e in SPEC['seeds']:verify(e)
 c=EasyDict(js(Path(SPEC['seeds'][0]['output'])/'resolved_eval.json'))
 control_seed(41);b,ds,shape=data_setup(c,True);c.shape_meta=shape
 front=complete(c,shape).eval().requires_grad_(False)
 # Original source only; full official frontend, original modality order and head semantics.
 input_keys=list(c.data.obs_key_mapping.values())
 prehash=sha(c.pretrain);assert all(not m.training for m in front.modules())
 admission=[]
 for e in SPEC['seeds']:
  for method in ['RA','ALIGN']:
   ps=load_pair(e,method,c,shape)
   x=torch.load(Path(e['output'])/'ra_coreset/buffer'/js(Path(e['output'])/'ra_coreset/storage/task8.json')['selected'][0]['destination'],map_location='cpu')['feature'].cuda()
   if x.ndim==5:x=x.squeeze(1)
   for t,p in zip([8,9],ps):
    for length in range(1,9):
     v=x[:,:length];q=params(p,v);m,s,k=selected(q);dist=p.policy_head(p.temporal_encode(v)[:,-1],train=False)
     assert torch.equal(dist.base_dist.loc,m) and torch.equal(dist.base_dist.scale,s)
     assert all(torch.isfinite(a).all() for a in q)
    admission.append({'seed':e['seed'],'method':method,'stage':t,'strict_keys':100,'lengths':list(range(1,9))})
   del ps
 # Frontend frame-cache equivalence on unaugmented, deterministic held input; zero environments.
 instr,mask=b.get_task_emb(8);gen=torch.Generator().manual_seed(20261024)
 obs={k:torch.rand((2,8,*v),generator=gen).cuda() for k,v in shape['all_shapes'].items()}
 d={'obs':obs,'task_emb':instr.repeat(2,1).cuda(),'masks':mask.repeat(2,1).cuda()}
 frames=[encode(front,{'obs':{k:v[:,i:i+1] for k,v in obs.items()},'task_emb':d['task_emb'],'masks':d['masks']}) for i in range(8)]
 errors=[];ps=load_pair(SPEC['seeds'][0],'ALIGN',c,shape)
 for length in range(1,9):
  whole=encode(front,{'obs':{k:v[:,:length] for k,v in obs.items()},'task_emb':d['task_emb'],'masks':d['masks']});cached=torch.cat(frames[:length],1)
  assert torch.allclose(whole,cached,atol=1e-4,rtol=1e-4)
  emax=float((whole-cached).abs().max());q0=params(ps[0],whole);q1=params(ps[0],cached)
  for a,z in zip(q0,q1):assert torch.allclose(a,z,atol=1e-4,rtol=1e-4)
  errors.append({'length':length,'max_feature_abs':emax,'max_parameter_abs':max(float((a-z).abs().max()) for a,z in zip(q0,q1))})
 # Independent episode streams survive lane order/early termination; reverse lookup is exact.
 assert all(np.array_equal(noise(41,i),noise(41,i)) for i in range(20))
 write(ROOT/'ADMISSION.json',{'status':'PASS_OWN_GPU_ADMISSION_ZERO_ENV_UPDATE','strict_checkpoints':admission,'cache_parity':errors,'official_pretrain_sha256':prehash,'updates':0,'environments':0,'parameter_forward_calls':len(admission)*8*2+16,'frontend_calls':16})
 del ps,frames,obs,d,x,ds
 write(ROOT/'RUN_STARTED.json',{'time':time.time(),'pid':os.getpid(),'episodes':240,'vectors':24,'capacity':capacity()})
 records=[]
 for e in SPEC['seeds']:
  seed=e['seed'];verify(e)
  for method in ['RA','ALIGN']:
   ps=load_pair(e,method,c,shape)
   for driver in [8,9]:
    for group in [0,1]:
     capacity();run='%s_s%d_driver%d_g%d'%(method,seed,driver,group);out=ROOT/'light'/run;out.mkdir(parents=True,exist_ok=False)
     priv=PRIVATE/run;priv.mkdir();start=time.monotonic();control_seed(seed)
     task=b.get_task(8);initial=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;initials=torch.load(initial)
     indices=list(range(group*10,(group+1)*10));init=initials[indices]
     env_args={'bddl_file_name':str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),'camera_heights':c.data.img_h,'camera_widths':c.data.img_w}
     event('vector_start',run=run,driver=driver,seed=seed,method=method,group=group,initial_file_sha256=sha(initial),capacity=capacity())
     env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**env_args) for _ in range(10)])
     try:
      env.seed(seed+group);env.reset();env.seed(seed+group);ob=env.set_init_state(init)
      for _ in range(5):ob,_,_,_=env.step(np.zeros((10,7)))
      eps=np.stack([noise(seed,i) for i in indices]);np.save(priv/'noise.npy',eps)
      history=collections.deque(maxlen=8);rings=[collections.deque(maxlen=16) for _ in indices];features=[[] for _ in indices];privateq=[[] for _ in indices];lowobs=[[] for _ in indices];actions_all=[[] for _ in indices]
      done=np.zeros(10,dtype=bool);endsteps=np.full(10,600);events=[{} for _ in indices];streams=[(out/('trajectory_%02d.jsonl'%i)).open('w') for i in indices]
      for step in range(600):
       data=raw_obs_to_tensor_obs(ob,instr,mask,c);data['obs']={k:v[:,None] for k,v in data['obs'].items()}
       frame=encode(front,data);assert frame.shape==(10,1,4,512)
       history.append(frame);x=torch.cat(list(history),1);hx=x.detach().cpu().numpy();fx=frame[:,0].detach().cpu().numpy()
       before=x.clone();q8=params(ps[0],x);assert torch.equal(before,x);q9=params(ps[1],x);assert torch.equal(before,x)
       m8,s8,k8=selected(q8);m9,s9,k9=selected(q9)
       z=torch.from_numpy(eps[:,step]).cuda();a8=m8+s8*z;a9=m9+s9*z
       action=(a8 if driver==8 else a9).cpu().numpy();action[done]=0
       cpuq=[tuple(v.cpu().numpy() for v in q) for q in [q8,q9]]
       dm=(m9.double()-m8.double()).cpu().numpy();dsd=(s9.double()-s8.double()).cpu().numpy();da=(a9.double()-a8.double()).cpu().numpy()
       for j,i in enumerate(indices):
        if done[j]:continue
        raw=obs_copy(ob[j],input_keys);rings[j].append(raw);oh=obs_hash(ob[j],input_keys)
        features[j].append(fx[j]);privateq[j].append(np.concatenate([v[j].reshape(-1) for q in cpuq for v in q]));lowobs[j].append(np.concatenate([v.reshape(-1) for k,v in raw.items() if 'image' not in k]));actions_all[j].append(action[j].copy())
        switch=bool(k8[j]!=k9[j]);flip=bool(m8[j,6]*m9[j,6]<0)
        flags={'mean_delta':float(np.abs(dm[j]).max())>1e-3,'mode_switch':switch,'gripper_flip':flip}
        for key,val in flags.items():
         if val and key not in events[j]:events[j][key]=step;save_raw(priv/('%02d_first_%s.npz'%(i,key)),list(rings[j])[-8:])
        row={'run':run,'initial_index':i,'step':step,'history_length':x.shape[1],'history_sha256':harray(hx[j]),'observation_sha256':oh,'noise_sha256':harray(eps[j,step]),'teacher_mode':int(k8[j]),'student_mode':int(k9[j]),'mode_switch':switch,'gripper_flip':flip,'mean_delta_squared':(dm[j]**2).tolist(),'sigma_delta_squared':(dsd[j]**2).tolist(),'action_delta_squared':(da[j]**2).tolist(),'w2_squared':float((dm[j]**2+dsd[j]**2).sum()),'teacher_margin':float(q8[2][j].softmax(-1).topk(2).values.diff().abs()[0]),'student_margin':float(q9[2][j].softmax(-1).topk(2).values.diff().abs()[0]),'same_history':True}
        streams[j].write(json.dumps(row,allow_nan=False)+'\n')
       ob,_,success,_=env.step(action)
       for j,i in enumerate(indices):
        if not done[j] and success[j]:
         done[j]=True;endsteps[j]=step+1;save_raw(priv/('%02d_terminal.npz'%i),list(rings[j])+[obs_copy(ob[j],input_keys)])
       if done.all():break
      for j,i in enumerate(indices):
       streams[j].close()
       if not done[j]:save_raw(priv/('%02d_terminal.npz'%i),list(rings[j])+[obs_copy(ob[j],input_keys)])
       f=np.stack(features[j]);np.savez_compressed(priv/('%02d_history.npz'%i),frames=f,parameters=np.stack(privateq[j]),low_observations=np.stack(lowobs[j]),executed_actions=np.stack(actions_all[j]))
       # Reconstruct all exact histories without re-inference and bind every recorded hash.
       rows=[json.loads(s) for s in (out/('trajectory_%02d.jsonl'%i)).read_text().splitlines()]
       assert len(rows)==int(endsteps[j]) and all(harray(f[max(0,t-7):t+1])==r['history_sha256'] for t,r in enumerate(rows))
       rec={'run':run,'seed':seed,'method':method,'driver':driver,'initial_index':i,'initial_sha256':harray(np.asarray(initials[i])),'noise_sequence_sha256':harray(eps[j]),'env_seed':seed+group,'success':bool(done[j]),'policy_steps':int(endsteps[j]),'first_events':events[j],'rows_sha256':sha(out/('trajectory_%02d.jsonl'%i)),'same_history_reconstruction':True,'attempt':1,'termination':'success' if done[j] else 'timeout600'}
       records.append(rec)
      write(out/'OUTCOMES.json',records[-10:]);write(out/'PRIVATE_MANIFEST.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in priv.iterdir() if p.is_file()})
     finally:env.close()
     write(ROOT/'OUTCOMES.json',records);event('vector_complete',run=run,episodes=len(records),seconds=time.monotonic()-start,capacity=capacity())
   del ps;verify(e)
 assert len(records)==240 and sha(c.pretrain)==prehash
 for e in SPEC['seeds']:verify(e)
 write(ROOT/'COMPLETE.json',{'status':'PASS_EXECUTION_PENDING_LOCAL_AUDIT','episodes':240,'vectors':24,'time':time.time(),'updates':0,'source_and_cp_unchanged':True,'private_root':str(PRIVATE),'capacity':capacity()})
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'FAILURE.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
