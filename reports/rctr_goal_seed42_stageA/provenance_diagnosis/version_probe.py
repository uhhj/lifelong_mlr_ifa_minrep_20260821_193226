"""Fixed two-version engineering comparison. No model, optimizer, or policy rollout."""
import os,sys,json,copy,time,traceback
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
sys.path.insert(0,str(W/"restoration_repair"))
import integration_probe as ip
p=ip.p
import mujoco,numpy as np,h5py
BASE=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis")
V=mujoco.__version__;assert V in ("3.2.3","2.3.5")
O=BASE/("mujoco_"+V)
def write(n,x):p.write(O/n,x)
def ledger(event,**kw):
 with (O/"attempts.jsonl").open("a") as f:
  f.write(json.dumps(dict(time=time.time(),event=event,**kw))+"\n");f.flush();os.fsync(f.fileno())
def main():
 O.mkdir(exist_ok=False)
 a=json.loads((W/"ASSETS.json").read_text());files=dict(a["input_hashes"])
 for path in (Path(__file__),W/"provenance_diagnosis/DIAGNOSTIC_PROTOCOL.md",
              W/"restoration_repair/integration_probe.py",W/"restoration_repair/repair_probe.py"):
  files[str(path)]=p.old.sha(path)
 write("CONTRACT.json",dict(version=V,library=mujoco.__file__,files=files,
  thresholds=dict(state=1e-5,observations=1e-6,repeat=1e-10),policy_continuations=0,optimizer_calls=0))
 p.log=ledger
 env=None;rows=[]
 try:
  c=p.old.cfg("eval");b=p.old.get_benchmark(c.benchmark_name)(c.data.task_order_index);task=b.get_task(6)
  ledger("construct_start")
  env=p.old.OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
      camera_heights=c.data.img_h,camera_widths=c.data.img_w)
  ledger("construct_complete")
  with h5py.File(a["hdf5"],"r") as f:
   for sel in a["chosen"]:
    d=sel["demo_id"];t=sel["target"];g=f["data"][d]
    states=g["states"][:];actions=g["actions"][:];xml=p.relocate(g.attrs["model_file"])
    p.reset(env,xml,states[0],d)
    row=dict(demo_id=d,target=t,prefix=[],snapshot_trials=[])
    rows.append(row)
    for i in range(t+1):
     env.set_state(states[i]);env.sim.forward();env.robots[0].controller.update(force=True)
     obs=p.step(env,actions[i],d+"_forced_prefix",i)
     row["prefix"].append(ip.state_error(env,env.get_sim_state(),states[i+1]))
    row["observation"]=p.old.compare_obs(obs,g,t)
    # Pure forward recomputes derived coordinates without integrating.
    row["eef_returned_vs_recorded"]=p.old.err(obs["robot0_eef_pos"],g["obs/ee_pos"][t])
    before=env.get_sim_state().copy()
    env.sim.forward();env._update_observables(force=True)
    fresh=env.env._get_observations()
    row["eef_after_forward_vs_recorded"]=p.old.err(fresh["robot0_eef_pos"],g["obs/ee_pos"][t])
    row["forward_physical_state_change"]=p.old.err(before,env.get_sim_state())
    snap=ip.snapshot(env);outs=[]
    for rep in range(2):
     ip.restore(env,snap)
     o=p.step(env,actions[t+1],d+"_repeat",t+1)
     st=env.get_sim_state().copy();outs.append((st,o))
     row["snapshot_trials"].append(ip.state_error(env,st,states[t+2]))
    row["repeat_error"]=p.old.err(outs[0][0],outs[1][0])
    row["repeat_obs_error"]=max(p.old.err(outs[0][1][k],outs[1][1][k]) for k in
     ("robot0_joint_pos","robot0_gripper_qpos","agentview_image","robot0_eye_in_hand_image"))
    # Controls are generated once, and replayed verbatim by the other engine.
    p.reset(env,xml,states[t],d+"_fixed_ctrl")
    ctrlpath=BASE/("private_controls_"+d+".npz")
    count=int(env.env.control_timestep/env.env.model_timestep)
    if V=="3.2.3":
     controls=[];orig=env.sim.step
     def capture():
      controls.append(env.sim.data.ctrl.copy());return orig()
     env.sim.step=capture
     try:p.step(env,actions[t],d+"_ctrl_capture",t)
     finally:env.sim.step=orig
     assert len(controls)==count
     np.savez(ctrlpath,controls=np.array(controls),post=env.get_sim_state().copy())
     row["fixed_controls"]=dict(substeps=count,sha256=p.old.sha(ctrlpath),
      historical=ip.state_error(env,env.get_sim_state(),states[t+1]))
    else:
     z=np.load(ctrlpath);controls=z["controls"];assert len(controls)==count
     ledger("step_start",label=d+"_fixed_ctrl",index=t)
     for ctrl in controls:
      env.sim.forward();env.sim.data.ctrl[:]=ctrl;env.sim.step()
     ledger("step_complete",label=d+"_fixed_ctrl",index=t)
     row["fixed_controls"]=dict(substeps=count,sha256=p.old.sha(ctrlpath),
      historical=ip.state_error(env,env.get_sim_state(),states[t+1]),
      vs_3_2_3=ip.state_error(env,env.get_sim_state(),z["post"]))
    write("RESULT.json",dict(status="PARTIAL",version=V,rows=rows))
    print(d,V,"target_error",row["prefix"][-1]["max_error"],
      "repeat",row["repeat_error"],"fixed",row["fixed_controls"],flush=True)
  for path,h in files.items():assert p.old.sha(path)==h,path
  write("RESULT.json",dict(status="DIAGNOSTIC_COMPLETE_NOT_ADMISSION",version=V,rows=rows,
   inputs_unchanged=True,policy_continuations=0,optimizer_calls=0))
 except BaseException as e:
  ledger("exception",error=repr(e));write("FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  if env is not None:env.close();ledger("closed")
if __name__=="__main__":main()
