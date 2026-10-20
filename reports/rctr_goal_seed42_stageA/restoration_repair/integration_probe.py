"""Preserve integrator/controller history, diagnose remaining errors, test full snapshots."""
import os,sys,json,copy,hashlib,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import repair_probe as p
import numpy as np,h5py,mujoco,random
O=p.O/"integration_probe"
def write(name,x):p.write(O/name,x)
def ledger(event,**kw):
 with (O/"attempts.jsonl").open("a") as f:
  f.write(json.dumps(dict(time=time.time(),event=event,**kw))+"\n");f.flush();os.fsync(f.fileno())
def state_error(env,pred,ref):
 dif=np.abs(pred-ref);nq=env.sim.model.nq
 top=np.argsort(dif[1:])[-8:][::-1]+1;items=[]
 for idx in top:
  vel=idx>nq;local=idx-(1+nq if vel else 1)
  adr=env.sim.model.jnt_dofadr if vel else env.sim.model.jnt_qposadr
  j=int(np.where(adr<=local)[0][-1])
  items.append(dict(kind="qvel" if vel else "qpos",joint=env.sim.model.joint_id2name(j),
    component=int(local-adr[j]),abs_error=float(dif[idx])))
 return dict(max_error=float(dif[1:].max()),qpos_error=float(dif[1:1+nq].max()),
  qvel_error=float(dif[1+nq:].max()),largest=items)
def snapshot(env):
 e=env.env;r=env.robots[0]
 state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
 integration=np.empty(mujoco.mj_stateSize(env.sim.model._model,state_spec))
 mujoco.mj_getState(env.sim.model._model,env.sim.data._data,integration,state_spec)
 return dict(integration=integration,state_spec=state_spec,
  controller=copy.deepcopy({k:v for k,v in r.controller.__dict__.items() if k!="sim"}),
  gripper=r.gripper.current_action.copy(),
  robot=copy.deepcopy({k:v for k,v in r.__dict__.items() if k.startswith("recent_") or k=="torques"}),
  env=copy.deepcopy({k:e.__dict__[k] for k in ("_obs_cache","_observables","cur_time","timestep","done")}),
  rng_python=random.getstate(),rng_numpy=np.random.get_state())
def restore(env,s):
 # Official integration state includes warmstart, controls, applied forces,
 # mocap and user state. Recompute derived fields, then restore observable caches.
 mujoco.mj_setState(env.sim.model._model,env.sim.data._data,s["integration"],s["state_spec"])
 env.sim.forward()
 r=env.robots[0]
 for k,v in s["controller"].items():r.controller.__dict__[k]=copy.deepcopy(v)
 r.gripper.current_action=s["gripper"].copy()
 for k,v in s["robot"].items():r.__dict__[k]=copy.deepcopy(v)
 for k,v in s["env"].items():env.env.__dict__[k]=copy.deepcopy(v)
 random.setstate(s["rng_python"]);np.random.set_state(s["rng_numpy"])
def main():
 O.mkdir(exist_ok=False)
 assert json.loads((p.O/"REPAIR_DIAGNOSTIC.json").read_text())["status"]=="DIAGNOSTIC_COMPLETE_NOT_BRANCH_ADMISSION"
 a=json.loads((p.W/"ASSETS.json").read_text())
 files=dict(a["input_hashes"])
 for x in (Path(__file__),p.R/"repair_probe.py",p.R/"RECOVERY_PROTOCOL.md"):files[str(x)]=p.old.sha(x)
 write("CONTRACT.json",dict(files=files,scope="engineering only, no policy or optimization",
  thresholds=dict(observations=1e-6,recorded_state=1e-5,repeat=1e-10)))
 env=None;rows=[]
 # Imported wrappers write their own reset bookkeeping in the first repair ledger.
 # Override logger locally so previous evidence remains immutable.
 p.log=lambda event,**kw:ledger(event,**kw)
 try:
  c=p.old.cfg("eval");b=p.old.get_benchmark(c.benchmark_name)(c.data.task_order_index);task=b.get_task(6)
  ledger("construct_start")
  env=p.old.OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
      camera_heights=c.data.img_h,camera_widths=c.data.img_w)
  ledger("construct_complete")
  with h5py.File(a["hdf5"],"r") as f:
   for sel in a["chosen"]:
    d=sel["demo_id"];t=sel["target"];g=f["data"][d];states=g["states"][:];actions=g["actions"][:]
    xml=p.relocate(g.attrs["model_file"])
    p.reset(env,xml,states[0],d)
    row=dict(demo_id=d,target=t,forced_prefix=[],snapshot_trials=[])
    rows.append(row)
    for i in range(t+1):
     # Restore only recorded physical coordinates; carry controller goals, actuator
     # commands, integrator warmstart and observable timing through the real prefix.
     env.set_state(states[i]);env.sim.forward();env.robots[0].controller.update(force=True)
     obs=p.step(env,actions[i],d+"_forced_prefix",i)
     rec=state_error(env,env.get_sim_state(),states[i+1]);rec["index"]=i
     row["forced_prefix"].append(rec)
    row["actual_target_observation"]=p.old.compare_obs(obs,g,t)
    # Actual policy target observation t corresponds to recorded state t+1.
    # Snapshots are tested as an engineering property, even if historical fidelity fails.
    snap=snapshot(env);pre=env.get_sim_state().copy();outs=[]
    for rep in range(2):
     restore(env,snap)
     restored=env.get_sim_state().copy()
     out=p.step(env,actions[t+1],d+"_snapshot_rep"+str(rep),t+1)
     post=env.get_sim_state().copy()
     outs.append((post,out))
     row["snapshot_trials"].append(dict(rep=rep,restore_error=p.old.err(restored,pre),
        historical_transition=state_error(env,post,states[t+2])))
    row["snapshot_repeat_error"]=p.old.err(outs[0][0],outs[1][0])
    row["snapshot_observation_repeat_error"]=max(p.old.err(outs[0][1][k],outs[1][1][k])
      for k in ("robot0_joint_pos","robot0_gripper_qpos","agentview_image","robot0_eye_in_hand_image"))
    write("RESULT.json",dict(status="PARTIAL",rows=rows))
    print(d,"last recorded transition",row["forced_prefix"][-1],
        "snapshot repeat",row["snapshot_repeat_error"],flush=True)
  for name,h in files.items():assert p.old.sha(name)==h,name
  write("RESULT.json",dict(status="ENGINEERING_DIAGNOSTIC_COMPLETE",rows=rows,inputs_unchanged=True,
    policy_continuations=0,optimizer_calls=0))
 except BaseException as e:
  ledger("exception",error=repr(e));write("FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),rows=rows));raise
 finally:
  if env is not None:env.close();ledger("closed")
if __name__=="__main__":main()
