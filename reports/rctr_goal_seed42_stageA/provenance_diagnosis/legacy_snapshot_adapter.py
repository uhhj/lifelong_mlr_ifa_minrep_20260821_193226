"""External MuJoCo 2.3.5 snapshot API compatibility, without numerical changes."""
import copy,random,sys
import version_probe as v
import numpy as np
assert v.V=="2.3.5"
v.O=v.BASE/"mujoco_2.3.5_legacy_snapshot_recovery"
def snapshot(env):
 e=env.env;r=env.robots[0];d=env.sim.data._data
 # 2.3.5 has no Python mj_getState/mjtState. Explicitly preserve the
 # integration inputs, including warmstart. No inferred historical values.
 fields=("qpos","qvel","act","qacc_warmstart","ctrl","qfrc_applied","xfrc_applied",
         "mocap_pos","mocap_quat","userdata")
 return dict(time=d.time,data={k:getattr(d,k).copy() for k in fields},
  controller=copy.deepcopy({k:x for k,x in r.controller.__dict__.items() if k!="sim"}),
  gripper=r.gripper.current_action.copy(),
  robot=copy.deepcopy({k:x for k,x in r.__dict__.items() if k.startswith("recent_") or k=="torques"}),
  env=copy.deepcopy({k:e.__dict__[k] for k in ("_obs_cache","_observables","cur_time","timestep","done")}),
  rng_python=random.getstate(),rng_numpy=np.random.get_state())
def restore(env,s):
 d=env.sim.data._data;d.time=s["time"]
 for k,x in s["data"].items():getattr(d,k)[:]=x
 env.sim.forward();r=env.robots[0]
 for k,x in s["controller"].items():r.controller.__dict__[k]=copy.deepcopy(x)
 r.gripper.current_action=s["gripper"].copy()
 for k,x in s["robot"].items():r.__dict__[k]=copy.deepcopy(x)
 for k,x in s["env"].items():env.env.__dict__[k]=copy.deepcopy(x)
 random.setstate(s["rng_python"]);np.random.set_state(s["rng_numpy"])
v.ip.snapshot=snapshot
v.ip.restore=restore
if __name__=="__main__":
 import json
 from pathlib import Path
 dst=v.BASE/"LEGACY_RECOVERY_ADDENDUM.json"
 assert not dst.exists()
 dst.write_text(json.dumps(dict(reason="MuJoCo 2.3.5 lacks Python mjtState API",
  original_exit=int((v.BASE/"probe235.exit").read_text()),
  adapter_sha256=v.p.old.sha(Path(__file__)),
  failed_attempt_ledger_sha256=v.p.old.sha(v.BASE/"mujoco_2.3.5/attempts.jsonl"),
  state_atol=1e-5,observation_atol=1e-6,repeat_atol=1e-10,
  extra_engineering_steps_due_to_failed_interface=47,
  no_policy_no_training=True),indent=2)+"\n")
 v.main()
