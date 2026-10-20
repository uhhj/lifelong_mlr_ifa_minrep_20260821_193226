"""Controller/cache/time-alignment repair diagnostics. No policy or optimizer."""
import os
os.environ["MUJOCO_GL"]="egl";os.environ["EGL_PLATFORM"]="surfaceless"
import sys,json,hashlib,time,traceback,copy
from pathlib import Path
import numpy as np,h5py
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
sys.path.insert(0,str(W))
import state_probe as old
R=W/"restoration_repair"
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/restoration_repair")
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def log(event,**kw):
 with (O/"attempts.jsonl").open("a") as f:
  f.write(json.dumps(dict(time=time.time(),event=event,**kw),allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def relocate(xml):
 tree=old.ET.fromstring(xml)
 for node in tree.iter():
  p=node.get("file")
  if p and "/chiliocosm/assets/" in p:
   node.set("file",str((old.S/"libero/libero/assets"/p.split("/chiliocosm/assets/",1)[1]).resolve()))
 return old.relocate(old.ET.tostring(tree,encoding="unicode"))[0]
def reset(env,xml,state,label):
 log("reset_start",label=label)
 env.seed(42);env.reset_from_xml_string(xml)
 env.sim.reset()
 env.set_state(state);env.sim.forward()
 env.env._obs_cache={}
 for obs in env.env._observables.values():obs.reset()
 env.robots[0].gripper.current_action=np.zeros(2)
 env.robots[0].controller.update(force=True)
 env._update_observables(force=True)
 log("reset_complete",label=label)
 return env.env._get_observations()
def step(env,act,label,index):
 log("step_start",label=label,index=index)
 obs,*_=env.step(act)
 log("step_complete",label=label,index=index)
 return obs
def main():
 O.mkdir(exist_ok=False)
 a=json.loads((W/"ASSETS.json").read_text())
 files=dict(a["input_hashes"])
 for name in ("state_probe.py","PROTOCOL.md"):files[str(W/name)]=old.sha(W/name)
 for p in R.glob("*"):
  if p.suffix in (".py",".md"):files[str(p)]=old.sha(p)
 write(O/"CONTRACT.json",dict(files=files,scientific_continuations=0,optimizer_calls=0))
 c=old.cfg("eval");b=old.get_benchmark(c.benchmark_name)(c.data.task_order_index);task=b.get_task(6)
 env=None;rows=[]
 try:
  log("construct_start")
  env=old.OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
    camera_heights=c.data.img_h,camera_widths=c.data.img_w)
  log("construct_complete")
  with h5py.File(a["hdf5"],"r") as f:
   for sel in a["chosen"]:
    d=sel["demo_id"];g=f["data"][d];t=sel["target"];states=g["states"][:];actions=g["actions"][:]
    xml=relocate(g.attrs["model_file"]);row=dict(demo_id=d,target=t,corrected_observations=[],direct=[],prefix=[])
    rows.append(row)
    # Dataset observation index i is the returned observation after action i.
    for i in range(t-7,t+1):
     obs=reset(env,xml,states[i+1],d+"_aligned_obs_"+str(i))
     row["corrected_observations"].append(dict(index=i,state_index=i+1,obs=old.compare_obs(obs,g,i)))
    for reconstruct in (False,True):
     outputs=[]
     for rep in range(2):
      label=d+"_reconstruct%d_rep%d"%(reconstruct,rep)
      reset(env,xml,states[0] if reconstruct else states[t],label)
      ctrl=env.robots[0].controller
      if reconstruct:
       for i in range(t):
        env.set_state(states[i]);env.sim.forward();ctrl.update(force=True)
        ctrl.set_goal(actions[i,:6])
        for substep in range(int(env.env.control_timestep/env.env.model_timestep)):
         env.robots[0].gripper.format_action(actions[i,6:])
       env.set_state(states[t]);env.sim.forward();ctrl.update(force=True)
      before=dict(controller_joint_vs_sim=old.err(ctrl.joint_pos,env.sim.data.qpos[ctrl.qpos_index]),
        goal_orientation_vs_actual=float(np.max(np.abs(ctrl.goal_ori-ctrl.ee_ori_mat))) if ctrl.goal_ori is not None else None,
        gripper_command_hash=hashlib.sha256(env.robots[0].gripper.current_action.tobytes()).hexdigest())
      obs=step(env,actions[t],label,t);after=env.get_sim_state().copy();outputs.append(after)
      nq=env.sim.model.nq
      row["direct"].append(dict(reconstruct=reconstruct,rep=rep,before=before,
        state_error=old.err(after[1:],states[t+1,1:]),
        qpos_error=old.err(after[1:1+nq],states[t+1,1:1+nq]),
        qvel_error=old.err(after[1+nq:],states[t+1,1+nq:]),
        observation=old.compare_obs(obs,g,t)))
     row["direct"][-1]["repeat_error"]=old.err(outputs[0],outputs[1])
    # Fresh synced prefix, fixed actions[0:t+1], compared at every transition.
    label=d+"_synced_prefix";reset(env,xml,states[0],label);errors=[];qerr=[]
    for i in range(t+1):
     obs=step(env,actions[i],label,i);after=env.get_sim_state()
     errors.append(old.err(after[1:],states[i+1,1:]));qerr.append(old.err(after[1:1+env.sim.model.nq],states[i+1,1:1+env.sim.model.nq]))
    row["prefix"]=dict(steps=t+1,max_error=max(errors),final_error=errors[-1],
      max_qpos_error=max(qerr),per_step_errors=errors,observation=old.compare_obs(obs,g,t))
    write(O/"REPAIR_DIAGNOSTIC.json",dict(status="PARTIAL",rows=rows))
    print(d,"direct",[(x["reconstruct"],x["state_error"],x.get("repeat_error")) for x in row["direct"]],
      "prefix",row["prefix"]["max_error"],flush=True)
  for p,h in files.items():assert old.sha(p)==h,p
  write(O/"REPAIR_DIAGNOSTIC.json",dict(status="DIAGNOSTIC_COMPLETE_NOT_BRANCH_ADMISSION",rows=rows,inputs_unchanged=True))
 except BaseException as e:
  log("exception",error=repr(e));write(O/"FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),rows=rows));raise
 finally:
  if env is not None:env.close();log("closed")
if __name__=="__main__":main()
