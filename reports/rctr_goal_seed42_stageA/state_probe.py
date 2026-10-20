"""Bounded physical restoration admission; no policy rollout or optimization."""
import os
os.environ["MUJOCO_GL"]="egl"
os.environ["EGL_PLATFORM"]="surfaceless"
os.environ["MUJOCO_EGL_DEVICE_ID"]="0"
import json,time,hashlib,traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import h5py
import robosuite
from coreset_exp.model import cfg
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020")
S=Path(os.environ["CORESET_SOURCE"])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
def log(event,**kw):
 with (O/"state_attempts.jsonl").open("a") as f:
  f.write(json.dumps(dict(event=event,time=time.time(),**kw),allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def err(a,b):return float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
def relocate(xml):
 root=ET.fromstring(xml);changes=[]
 for node in root.iter():
  p=node.get("file")
  if not p:continue
  parts=p.split("/")
  if "robosuite" in parts:
   j=max(i for i,x in enumerate(parts) if x=="robosuite")
   q=Path(robosuite.__file__).parent.joinpath(*parts[j+1:])
  elif "libero" in parts:
   j=max(i for i,x in enumerate(parts) if x=="libero")
   q=S/"libero"/"libero"/Path(*parts[j+1:])
  else:q=Path(p)
  assert q.exists(),str(q)
  node.set("file",str(q));changes.append(dict(old=p,new=str(q)))
 return ET.tostring(root,encoding="unicode"),changes
def compare_obs(obs,g,i):
 d={}
 for stored,native in [("joint_states","robot0_joint_pos"),("gripper_states","robot0_gripper_qpos"),
                      ("ee_pos","robot0_eef_pos")]:
  d[stored]=err(obs[native],g["obs"][stored][i])
 for stored,native in [("agentview_rgb","agentview_image"),("eye_in_hand_rgb","robot0_eye_in_hand_image")]:
  a=obs[native].astype(float);b=g["obs"][stored][i].astype(float)
  d[stored]=dict(mae=float(np.abs(a-b).mean()),max=err(a,b),
    flipped_mae=float(np.abs(a[::-1]-b).mean()),flipped_max=err(a[::-1],b))
 return d
def reset_to(env,xml,state,label):
 log("reset_start",label=label)
 env.seed(42);env.reset_from_xml_string(xml)
 obs=env.set_init_state(state)
 log("reset_complete",label=label)
 return obs
def step(env,action,label,index):
 log("engineering_step_start",label=label,index=index)
 obs,_,_,_=env.step(action)
 log("engineering_step_complete",label=label,index=index)
 return obs
def main():
 O.mkdir(exist_ok=False)
 a=json.loads((W/"ASSETS.json").read_text())
 files=dict(a["input_hashes"])
 for p in W.glob("*"):
  if p.is_file() and p.suffix in (".py",".md",".json"):files[str(p)]=sha(p)
 c=cfg("eval");bench=get_benchmark(c.benchmark_name)(c.data.task_order_index);task=bench.get_task(6)
 bddl=Path(c.bddl_folder)/task.problem_folder/task.bddl_file;files[str(bddl)]=sha(bddl)
 write(O/"RESTORATION_CONTRACT.json",dict(files=files,optimizer_calls=0,policy_rollouts=0,
   protocol_sha=sha(W/"PROTOCOL.md"),diagnostic_only=True))
 rows=[];env=None
 try:
  log("environment_construct_start")
  env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=c.data.img_h,camera_widths=c.data.img_w)
  log("environment_construct_complete")
  with h5py.File(a["hdf5"],"r") as f:
   for selected in a["chosen"]:
    d=selected["demo_id"];g=f["data"][d];t=selected["target"]
    states=g["states"][:];actions=g["actions"][:];xml,changes=relocate(g.attrs["model_file"])
    row=dict(demo_id=d,sample_id=selected["sample_id"],target=t,
       xml_original_sha=hashlib.sha256(g.attrs["model_file"].encode()).hexdigest(),
       xml_resolved_sha=hashlib.sha256(xml.encode()).hexdigest(),asset_relocations=changes,
       observation_checks=[],direct=[],prefix=[])
    rows.append(row)
    # Recover all eight actual native history observations. No actions executed here.
    for i in range(t-7,t+1):
     obs=reset_to(env,xml,states[i],d+"_observe_"+str(i))
     row["observation_checks"].append(dict(index=i,state_error=err(env.get_sim_state(),states[i]),obs=compare_obs(obs,g,i)))
    # Fixed two temporal conventions, two repetitions. No success-based choices.
    for offset in (0,1):
     outputs=[]
     for rep in (0,1):
      label=d+"_direct_offset%d_rep%d"%(offset,rep)
      reset_to(env,xml,states[t],label)
      obs=step(env,actions[t+offset],label,t+offset)
      state=env.get_sim_state().copy();outputs.append((state,obs))
      row["direct"].append(dict(offset=offset,rep=rep,
        next_state_error_without_time=err(state[1:],states[t+1,1:]),
        next_state_time_error=abs(float(state[0]-states[t+1,0])),
        next_observation=compare_obs(obs,g,t+1)))
     row["direct"][-1]["repeat_state_error"]=err(outputs[0][0],outputs[1][0])
     row["direct"][-1]["repeat_rgb_error"]=max(err(outputs[0][1][k],outputs[1][1][k])
       for k in ("agentview_image","robot0_eye_in_hand_image"))
    exact=[x for x in row["direct"] if x["next_state_error_without_time"]<=1e-5]
    # If hidden actuator/controller state cannot be reconstructed directly, replay prefix.
    if not exact:
     for offset in (0,1):
      label=d+"_prefix_offset%d"%offset
      obs=reset_to(env,xml,states[0],label)
      max_path_error=0.
      for i in range(t):
       obs=step(env,actions[i+offset],label,i+offset)
       max_path_error=max(max_path_error,err(env.get_sim_state()[1:],states[i+1,1:]))
      row["prefix"].append(dict(offset=offset,target_state_error_without_time=err(env.get_sim_state()[1:],states[t,1:]),
        max_path_state_error_without_time=max_path_error,target_observation=compare_obs(obs,g,t)))
    write(O/"STATE_PROBE.json",dict(status="PARTIAL",rows=rows))
    print(d,"direct errors",[r["next_state_error_without_time"] for r in row["direct"]],
      "prefix errors",[r["target_state_error_without_time"] for r in row["prefix"]],flush=True)
  # This checks only the first restoration gate; full branch snapshot/history gate is later.
  passed=all(any(r["next_state_error_without_time"]<=1e-5 for r in row["direct"]) or
    any(r["max_path_state_error_without_time"]<=1e-5 for r in row["prefix"]) for row in rows)
  for p,h in files.items():assert sha(p)==h,("Changed input",p)
  write(O/"STATE_PROBE.json",dict(status="TRANSITION_GATE_PASS" if passed else "BLOCKED_STATE_RESTORATION",
   rows=rows,inputs_unchanged=True,optimizer_calls=0,policy_rollouts=0,
   reason=None if passed else "Neither fixed direct-state convention nor fixed prefix convention reproduces recorded physical transitions within frozen tolerance. Do not start 54 continuations."))
 except BaseException as e:
  log("technical_exception",error=repr(e));write(O/"STATE_PROBE_FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),rows=rows));raise
 finally:
  if env is not None:env.close();log("environment_closed")
if __name__=="__main__":main()
