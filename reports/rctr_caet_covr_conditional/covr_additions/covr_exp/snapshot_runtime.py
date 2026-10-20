"""Native current-environment snapshot primitives inherited from audited A2 recovery.
No A2 data or model paths are reused. A new live environment is retained per initial.
"""
import os,copy,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
from libero.lifelong.metric import raw_obs_to_tensor_obs
O=None
COUNTS=None
START=None

def initialize(output):
 global O,COUNTS,START
 O=Path(output);O.mkdir(parents=True,exist_ok=True)
 COUNTS=dict(prefix_steps=0,engineering_steps=0,screening_steps=0,optimizer_calls=0,completed_branches=0)
 START=time.time()

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def write(name,obj):
 path=O/name;tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def ledger(event,**kw):
 with (O/"attempts.jsonl").open("a") as f:
  f.write(json.dumps(dict(event=event,unix=time.time(),**kw),allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())

def status(phase,**kw):write("RUN_STATUS.json",dict(status=phase,unix=time.time(),elapsed_seconds=time.time()-START,counts=COUNTS,**kw))

def cpu_rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])

def set_rng(r):
 random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2]);torch.cuda.set_rng_state_all(r[3])

def maxerr(a,b):
 aa=np.asarray(a,dtype=np.float64);bb=np.asarray(b,dtype=np.float64)
 assert aa.shape==bb.shape
 return float(np.max(np.abs(aa-bb))) if aa.size else 0.

def obs_error(a,b):
 assert set(a)==set(b)
 values={}
 for k in a:
  if isinstance(a[k],(np.ndarray,float,int,np.number,bool)):
   values[k]=maxerr(a[k],b[k])
  else:assert a[k]==b[k],k
 return max(values.values(),default=0.),values

def snapshot(env,policy,obs,history):
 r=env.robots[0];e=env.env
 # Full native MjData preserves derived observation/constraint phase as well as integration fields.
 return dict(data=copy.copy(env.sim.data._data),
  controller=copy.deepcopy({k:v for k,v in r.controller.__dict__.items() if k!="sim"}),
  gripper=r.gripper.current_action.copy(),
  robot=copy.deepcopy({k:v for k,v in r.__dict__.items() if k.startswith("recent_") or k=="torques"}),
  env=copy.deepcopy({k:e.__dict__[k] for k in ("_obs_cache","_observables","cur_time","timestep","done")}),
  env_rng=copy.deepcopy({k:e.__dict__[k] for k in ("np_random","_np_random") if k in e.__dict__}),
  rng=cpu_rng(),queue=copy.deepcopy(getattr(policy,"latent_queue",[])),obs=copy.deepcopy(obs),
  history=copy.deepcopy(history),physical=env.get_sim_state().copy())

def restore(env,policy,s):
 # Do not call forward or update observables here: both would change the captured phase.
 env.sim.data._data=copy.copy(s["data"])
 r=env.robots[0]
 for k,v in s["controller"].items():r.controller.__dict__[k]=copy.deepcopy(v)
 r.gripper.current_action=s["gripper"].copy()
 for k,v in s["robot"].items():r.__dict__[k]=copy.deepcopy(v)
 for k,v in s["env"].items():env.env.__dict__[k]=copy.deepcopy(v)
 for k,v in s["env_rng"].items():env.env.__dict__[k]=copy.deepcopy(v)
 set_rng(s["rng"]);policy.latent_queue=copy.deepcopy(s["queue"])
 err=maxerr(env.get_sim_state(),s["physical"]);assert err<=1e-10,("restore_state",err)
 return copy.deepcopy(s["obs"]),copy.deepcopy(s["history"]),err

def noise(seed,n=600):
 g=torch.Generator(device="cpu");g.manual_seed(seed)
 return torch.randn(n,7,generator=g).numpy()

def batch_of(history,instr,mask,c):
 data=[raw_obs_to_tensor_obs([o],instr,mask,c) for o in history]
 return dict(obs={k:torch.stack([v["obs"][k] for v in data],1) for k in data[0]["obs"]},
  task_emb=data[0]["task_emb"],masks=data[0]["masks"])

@torch.no_grad()
def params(policy,batch):
 d=policy.preprocess_input(batch,train_mode=False)
 z=policy.temporal_encode(policy.spatial_encode(d))[:,-1]
 m,s,l=policy.policy_head.forward_fn(z)
 dist=policy.policy_head(z,train=False);idx=l.argmax(-1)
 assert torch.equal(dist.base_dist.loc,m[torch.arange(m.shape[0],device=m.device),idx])
 assert torch.equal(dist.base_dist.scale,s[torch.arange(m.shape[0],device=m.device),idx])
 out=[v[0].detach().cpu().numpy().copy() for v in (m,s,l)]
 assert all(np.isfinite(v).all() for v in out)
 return out,int(idx[0])

def normal(par,idx,epsilon):return par[0][idx]+par[1][idx]*epsilon

def envstep(env,action,kind):
 if kind=="screening":write("LIVE_STEP.json",dict(event="start",index=COUNTS[kind+"_steps"],unix=time.time()))
 else:ledger("control_step_start",kind=kind,index=COUNTS[kind+"_steps"])
 o,_,done,_=env.step(action)
 COUNTS[kind+"_steps"]+=1
 if kind=="screening":write("LIVE_STEP.json",dict(event="complete",index=COUNTS[kind+"_steps"],unix=time.time()))
 else:ledger("control_step_complete",kind=kind,index=COUNTS[kind+"_steps"])
 return o,bool(done)

def stack(history,obs):return (history+[obs])[-8:]

def native_check(policy,hist,instr,mask,c):
 results=[]
 for n in (1,4,8):
  batch=batch_of(hist[-n:],instr,mask,c);r=cpu_rng()
  with torch.no_grad():native=policy.get_action(batch)[0]
  set_rng(r);p,idx=params(policy,batch)
  # Same device and native Normal.sample path for RNG equivalence, independent of paired tables.
  set_rng(r)
  with torch.no_grad():
   d=policy.preprocess_input(batch,train_mode=False)
   z=policy.temporal_encode(policy.spatial_encode(d))[:,-1]
   m,s,l=policy.policy_head.forward_fn(z);k=l.argmax(-1)
   dist=torch.distributions.Independent(torch.distributions.Normal(m[0,k[0]],s[0,k[0]]),1)
   custom=dist.sample().cpu().numpy()
  err=maxerr(native,custom);assert np.allclose(native,custom,atol=1e-5,rtol=1e-5),(n,err)
  results.append(dict(history=n,max_action_error=err))
  set_rng(r)
 return results

def gate(env,policy,snap,index,instr,mask,c):
 h=stack(snap["history"],snap["obs"])
 result=dict(initial_index=index,native_interface=native_check(policy,h,instr,mask,c))
 table=noise(42050000+index,8);traces=[]
 for rep in range(2):
  obs,history,err=restore(env,policy,snap);trace=[]
  ledger("engineering_repeat_start",initial_index=index,repeat=rep)
  for step in range(8):
   history=stack(history,obs);par,idx=params(policy,batch_of(history,instr,mask,c))
   action=normal(par,idx,table[step]);obs,done=envstep(env,action,"engineering")
   trace.append((env.get_sim_state().copy(),copy.deepcopy(obs),par,action.copy(),done))
  traces.append(trace)
 checks=[]
 for a,b in zip(*traces):
  oe,fields=obs_error(a[1],b[1]);pe=maxerr(a[0],b[0])
  ge=max(maxerr(x,y) for x,y in zip(a[2],b[2]));ae=maxerr(a[3],b[3])
  assert pe<=1e-10 and oe<=1e-10 and ge<=1e-5 and ae<=1e-5 and a[4]==b[4],(pe,oe,ge,ae)
  checks.append(dict(physical=pe,observations=oe,gmm=ge,action=ae,observation_fields=fields))
 result.update(status="PASS",repeat_checks=checks)
 ledger("engineering_gate_complete",initial_index=index)
 return result
