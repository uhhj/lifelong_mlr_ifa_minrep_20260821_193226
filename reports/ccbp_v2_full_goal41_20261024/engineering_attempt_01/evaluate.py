"""Exactly 800 paired episodes, no retries; no raw observations/actions exported."""
import collections
from runtime import *
from coreset_exp.model import data_setup,complete
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.metric import raw_obs_to_tensor_obs,SubprocVectorEnv,OffScreenRenderEnv
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def encode(front,d):
 x={'obs':{k:v.clone() for k,v in d['obs'].items()},'task_emb':d['task_emb'].clone(),'masks':d['masks'].clone()}
 return front.spatial_encode(front.preprocess_input(x,train_mode=False))
def load(arm,stage,c,shape):
 p=BriefPolicy(copy.deepcopy(c),shape).cuda().eval().requires_grad_(False)
 sd=torch.load(model_path(arm,stage),map_location='cpu')['state_dict'];sd={(k[7:] if k.startswith('module.') else k):v for k,v in sd.items()}
 assert len(sd)==len(p.state_dict())==100;p.load_state_dict(sd,strict=True)
 assert all(torch.equal(v.cpu(),sd[k]) for k,v in p.state_dict().items());return p
def event(kind,**kw):
 with (R/'EVAL_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n');f.flush()
@torch.no_grad()
def main():
 verify();capacity();assert not (R/'EVALUATION_STARTED.json').exists(),'No blind formal evaluation resume'
 c=EasyDict(js(O/'resolved_eval.json'));c.training=False;control_seed(41);b,ds,shape=data_setup(c,True);c.shape_meta=shape
 front=complete(c,shape).eval().requires_grad_(False);instr,mask=b.get_task_emb(8)
 gen=torch.Generator().manual_seed(20261024);obs={k:torch.rand((2,8,*v),generator=gen).cuda() for k,v in shape['all_shapes'].items()}
 base={'task_emb':instr.repeat(2,1).cuda(),'masks':mask.repeat(2,1).cuda()};frames=[encode(front,dict(obs={k:v[:,i:i+1] for k,v in obs.items()},**base)) for i in range(8)]
 parity=[];p=load('B',6,c,shape)
 for n in range(1,9):
  whole=encode(front,dict(obs={k:v[:,:n] for k,v in obs.items()},**base));cached=torch.cat(frames[:n],1)
  assert torch.allclose(whole,cached,atol=1e-4,rtol=1e-4)
  q0=p.policy_head.forward_fn(p.temporal_encode(whole)[:,-1]);q1=p.policy_head.forward_fn(p.temporal_encode(cached)[:,-1]);assert all(torch.allclose(x,y,atol=1e-4,rtol=1e-4) for x,y in zip(q0,q1))
  parity.append(dict(length=n,max_feature_abs=float((whole-cached).abs().max()),max_param_abs=max(float((x-y).abs().max()) for x,y in zip(q0,q1))))
 write(R/'EVAL_ADMISSION.json',dict(status='PASS',cache_parity=parity,frontend_calls=16,policy_parameter_calls=16,environments=0,updates=0,official_cp_sha256=sha(c.pretrain)))
 write(R/'EVALUATION_STARTED.json',dict(time=time.time(),pid=os.getpid(),budget=800));records=[]
 for arm in ARMS:
  for stage in range(6,10):
   p=load(arm,stage,c,shape);cp_sha=sha(model_path(arm,stage))
   for taskid in range(6,stage+1):
    instr,mask=b.get_task_emb(taskid);task=b.get_task(taskid);initial=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;initials=torch.load(initial);initial_sha=sha(initial)
    args=dict(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),camera_heights=c.data.img_h,camera_widths=c.data.img_w)
    for group in [0,1]:
     capacity();run='%s_s%d_t%d_g%d'%(arm,stage,taskid,group);indices=list(range(group*10,(group+1)*10));tic=time.monotonic()
     eps=np.stack([np.random.RandomState(20261024+41*1000+taskid*100+i).standard_normal((600,7)).astype(np.float32) for i in indices])
     event('start',run=run,cp_sha256=cp_sha,initial_file_sha256=initial_sha)
     env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**args) for _ in indices])
     try:
      control_seed(41);env.seed(41+group);env.reset();env.seed(41+group);ob=env.set_init_state(initials[indices])
      for _ in range(5):ob,_,_,_=env.step(np.zeros((10,7)))
      hist=collections.deque(maxlen=8);done=np.zeros(10,dtype=bool);end=np.full(10,600);hashers=[hashlib.sha256() for _ in indices]
      for step in range(600):
       data=raw_obs_to_tensor_obs(ob,instr,mask,c);data['obs']={k:v[:,None] for k,v in data['obs'].items()};hist.append(encode(front,data));x=torch.cat(list(hist),1)
       mu,sd,l=p.policy_head.forward_fn(p.temporal_encode(x)[:,-1]);k=l.argmax(-1);ix=torch.arange(10,device='cuda');act=(mu[ix,k]+sd[ix,k]*torch.from_numpy(eps[:,step]).cuda()).cpu().numpy();act[done]=0
       for j in range(10):
        if not done[j]:hashers[j].update(np.ascontiguousarray(act[j]).tobytes())
       ob,_,success,_=env.step(act)
       for j in range(10):
        if not done[j] and success[j]:done[j]=True;end[j]=step+1
       if done.all():break
      batch=[dict(arm=arm,stage=stage,task=taskid,initial_index=i,env_seed=41+group,initial_file_sha256=initial_sha,initial_sha256=ah(np.asarray(initials[i])),noise_sha256=ah(eps[j]),action_stream_sha256=hashers[j].hexdigest(),cp_sha256=cp_sha,success=bool(done[j]),steps=int(end[j]),attempt=1,termination='success' if done[j] else 'timeout600') for j,i in enumerate(indices)]
      records.extend(batch);write(R/'outcomes'/(run+'.json'),batch);write(R/'OUTCOMES.json',records)
     finally:env.close()
     event('complete',run=run,seconds=time.monotonic()-tic,episodes=len(records))
 assert len(records)==800;verify();write(R/'EVALUATION_COMPLETE.json',dict(status='PASS_EXECUTION_PENDING_LOCAL_AUDIT',episodes=800,vectors=80,cells=40,time=time.time()))
if __name__=='__main__':main()
