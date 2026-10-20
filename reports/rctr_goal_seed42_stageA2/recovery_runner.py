"""Frozen bounded current-state counterfactual screening. No optimizer or training."""
import os
os.environ["MUJOCO_GL"]="egl"
os.environ["EGL_PLATFORM"]="surfaceless"
import sys,copy,json,time,hashlib,traceback,random
from pathlib import Path
import numpy as np
import torch
import mujoco
from coreset_exp.model import cfg,data_setup,complete,strict_downstream
from libero.lifelong.metric import OffScreenRenderEnv,raw_obs_to_tensor_obs
from libero.lifelong.utils import control_seed

W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA2_20261020")
ORIGINAL=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020")
O=ORIGINAL/"engineering_recovery/dedicated_env_v1"
P=Path("/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018")
CP=P/"ra_coreset/models/mlrifa_6_model_ep99.pth"
S=Path(os.environ["CORESET_SOURCE"])
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
def candidates():
 yield dict(id="normal",axis=None,sign=0,recovery="original")
 for axis in range(3):
  for sign in (-1,1):
   for recovery in ["original",0,1,2,3,4]:
    yield dict(id="a%d_%s_r%s"%(axis,"p" if sign>0 else "m",recovery),axis=axis,sign=sign,recovery=recovery)
def branch(env,policy,snap,index,rep,cand,table,instr,mask,c):
 obs,history,_=restore(env,policy,snap)
 identity=dict(initial_index=index,repeat=rep,candidate=cand,noise_seed=42040000+100*index+rep,
  noise_sha256=digest(table))
 ledger("branch_start",**identity)
 action_hash=hashlib.sha256();state_hash=hashlib.sha256();amplitude=0.;recovery_delta=None;mode_dist=None
 begin=time.time();success=False;skip=None;steps=0
 for step in range(600):
  history=stack(history,obs);par,idx=params(policy,batch_of(history,instr,mask,c))
  action=normal(par,idx,table[step])
  if step==0 and cand["axis"] is not None:
   base=np.clip(action,-1.,1.);modified=base.copy()
   modified[cand["axis"]]=np.clip(base[cand["axis"]]+.05*cand["sign"],-1.,1.)
   amplitude=float(np.linalg.norm(modified-base))
   if amplitude<=1e-7:skip="SATURATED_ZERO_INTERVENTION";break
   action=modified
  if step==1:
   mode_dist=[float(np.linalg.norm(par[0][j]-par[0][idx])) for j in range(5)]
   if cand["axis"] is not None and cand["recovery"]!="original":
    replacement=par[0][cand["recovery"]].copy()
    recovery_delta=float(np.linalg.norm(np.clip(replacement,-1,1)-np.clip(action,-1,1)))
    action=replacement
  action_hash.update(action.tobytes())
  obs,success=envstep(env,action,"screening");steps+=1
  state_hash.update(env.get_sim_state().tobytes())
  if success:break
 row=dict(**identity,status="SKIPPED" if skip else "COMMITTED",skip_reason=skip,success=success if not skip else None,
  steps=steps,elapsed_seconds=time.time()-begin,intervention_l2=amplitude,recovery_action_l2=recovery_delta,
  component_mean_l2_from_argmax=mode_dist,action_trace_sha256=action_hash.hexdigest(),state_trace_sha256=state_hash.hexdigest())
 with (O/"episodes.jsonl").open("a") as f:
  f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
 COUNTS["completed_branches"]+=1
 ledger("branch_complete",**identity,status=row["status"],steps=steps,success=row["success"])
 status("SCREENING",initial_index=index,repeat=rep,candidate=cand["id"])
 print("COMMIT",index,rep,cand["id"],row["status"],success,steps,flush=True)

def main():
 assert not (O/"CONTRACT.json").exists(),"No blind partial restart"
 assert (ORIGINAL/"runner.exit").read_text().strip()=="1"
 old_contract=json.loads((ORIGINAL/"CONTRACT.json").read_text())
 for path,h in old_contract["files"].items():assert sha(path)==h,path
 expected={r["initial_index"]:r for r in json.loads((ORIGINAL/"SNAPSHOT_INVENTORY.json").read_text())}
 assert mujoco.__version__=="3.2.3"
 assert torch.cuda.is_available()
 torch.backends.cudnn.benchmark=True
 control_seed(42);c=cfg("eval")
 bench,datasets,shape=data_setup(c,True);c.shape_meta=shape
 task=bench.get_task(6);instr,mask=bench.get_task_emb(6)
 initial_path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file
 bddl=Path(c.bddl_folder)/task.problem_folder/task.bddl_file
 files=[CP,Path(c.pretrain),initial_path,bddl,P/"CONTRACT.json",P/"MODAL_CONTRACT.json",
        P/"resolved_eval.json",P/"ra_coreset/storage/task6.json"]
 files+=list(W.glob("*.py"))+list(W.glob("*.md"))+list(W.glob("*.sh"))
 files+=list((S/"libero/lifelong").rglob("*.py"))+list((S/"coreset_exp").glob("*.py"))
 hashes={str(p):sha(p) for p in files}
 write("CONTRACT.json",dict(files=hashes,created_unix=time.time(),mujoco=mujoco.__version__,
  torch=torch.__version__,gpu=torch.cuda.get_device_name(),scope="A2 only; no training",
  max_branches=333,max_engineering_steps=48,max_prefix_steps=135))
 write("resolved_eval.json",dict(c))
 policy=complete(c,shape);loaded=strict_downstream(policy,CP);policy.eval();policy.requires_grad_(False)
 states=torch.load(initial_path);assert len(states)>22
 write("MODEL_ADMISSION.json",dict(status="PASS",downstream_keys=loaded,parent_sha256=sha(CP),state=True))
 env=None;environments={};snapshots={};gates=[];inventory=[]
 try:
  for index in (20,21,22):
   ledger("environment_construct_start",initial_index=index)
   env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=c.data.img_h,camera_widths=c.data.img_w)
   environments[index]=env
   ledger("environment_construct_complete",initial_index=index)
   status("STATE_COLLECTION",initial_index=index)
   env.seed(42);env.reset();env.seed(42);policy.reset()
   obs=env.set_init_state(states[index]);history=[];already=False
   for _ in range(5):obs,d=envstep(env,np.zeros(7),"prefix");already=already or d
   table=noise(42030000+index,40)
   for step in range(40):
    history=stack(history,obs);par,idx=params(policy,batch_of(history,instr,mask,c))
    obs,d=envstep(env,normal(par,idx,table[step]),"prefix");already=already or d
   # history contains pre-step40 observation; current observation is one transition later.
   history=history[-7:]
   row=dict(initial_index=index,initial_sha256=digest(states[index]),prefix_noise_seed=42030000+index,
    prefix_noise_sha256=digest(table),prefix_success=already,eligible=not already,
    physical_sha256=digest(env.get_sim_state()),observation_hashes={k:digest(v) for k,v in obs.items() if isinstance(v,np.ndarray)},
    history_length=len(history)+1,control_frequency=float(env.env.control_freq),
    joint_norm=float(np.linalg.norm(obs["robot0_joint_pos"])),gripper_norm=float(np.linalg.norm(obs["robot0_gripper_qpos"])),
    contact_count=int(env.sim.data.ncon))
   for key in ("physical_sha256","observation_hashes","prefix_noise_sha256","initial_sha256","eligible"):
    assert row[key]==expected[index][key],("recollected_identity",index,key)
   row["recollected_identity"]="EXACT_MATCH_ORIGINAL"
   inventory.append(row)
   if not already:
    snapshots[index]=snapshot(env,policy,obs,history)
    # Gate all snapshots only after every environment has been constructed.
   write("SNAPSHOT_INVENTORY.json",inventory);write("ENGINEERING_GATE.json",dict(status="PARTIAL",states=gates))
  assert snapshots,"No eligible fixed snapshot; no replacement sampling allowed"
  for index,snap in snapshots.items():
   gates.append(gate(environments[index],policy,snap,index,instr,mask,c))
  assert COUNTS["engineering_steps"]<=48 and COUNTS["prefix_steps"]<=135
  write("ENGINEERING_GATE.json",dict(status="PASS",states=gates,eligible_states=list(snapshots)))
  # Source and parent identity must still match before any branch starts.
  for p,h in hashes.items():assert sha(p)==h,p
  for index,snap in snapshots.items():
   env=environments[index]
   for rep in range(3):
    table=noise(42040000+100*index+rep)
    for cand in candidates():branch(env,policy,snap,index,rep,cand,table,instr,mask,c)
  for p,h in hashes.items():assert sha(p)==h,p
  assert COUNTS["completed_branches"]==111*len(snapshots)<=333
  write("INPUT_PROTECTION.json",dict(status="PASS",files=hashes))
  write("EXECUTION_COMPLETE.json",dict(status="PASS",counts=COUNTS,eligible_states=list(snapshots),
   elapsed_seconds=time.time()-START,snapshot_storage="Three in-process full snapshots; no persistent weight/data copies"))
  status("AWAITING_ANALYSIS")
 except BaseException as e:
  ledger("technical_exception",error=repr(e))
  write("FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),counts=COUNTS))
  status("BLOCKED",error=repr(e));raise
 finally:
  for index,env in environments.items():
   env.close();ledger("environment_closed",initial_index=index)
if __name__=="__main__":main()
