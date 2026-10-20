"""Disentangle native continuation, integration restore, and exact coordinate injection."""
import sys,copy,json,traceback
from pathlib import Path
import legacy_snapshot_adapter as a
v=a.v;p=v.p;np=v.np;h5py=v.h5py
assert v.V=="2.3.5"
v.O=v.BASE/"controller_phase"
def main():
 v.O.mkdir(exist_ok=False);p.log=v.ledger
 assets=json.loads((v.W/"ASSETS.json").read_text())
 v.write("CONTRACT.json",dict(scope="engineering only; unchanged thresholds",files={
  str(Path(__file__)):p.old.sha(__file__),
  str(Path(a.__file__)):p.old.sha(a.__file__)},policy_continuations=0,optimizer_calls=0))
 c=p.old.cfg("eval");b=p.old.get_benchmark(c.benchmark_name)(c.data.task_order_index);task=b.get_task(6)
 env=None;rows=[]
 try:
  v.ledger("construct_start")
  env=p.old.OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
      camera_heights=c.data.img_h,camera_widths=c.data.img_w)
  v.ledger("construct_complete")
  with h5py.File(assets["hdf5"],"r") as f:
   for sel in assets["chosen"]:
    d=sel["demo_id"];t=sel["target"];g=f["data"][d]
    st=g["states"][:];act=g["actions"][:]
    p.reset(env,p.relocate(g.attrs["model_file"]),st[0],d)
    row=dict(demo_id=d,history=[],variants=[]);rows.append(row)
    for i in range(t+1):
     env.set_state(st[i]);env.sim.forward();env.robots[0].controller.update(force=True)
     ob=p.step(env,act[i],d+"_prefix",i)
     if i>=t-7:row["history"].append(dict(index=i,state=v.ip.state_error(env,env.get_sim_state(),st[i+1]),
                                      observation=p.old.compare_obs(ob,g,i)))
    snap=a.snapshot(env);full=copy.copy(env.sim.data._data)
    for variant in ("native","full_copy","integration","exact_coordinates","force_controller"):
     if variant!="native":
      a.restore(env,snap)
      if variant=="full_copy":env.sim.data._data=copy.copy(full)
      if variant=="exact_coordinates":
       env.set_state(st[t+1]);env.sim.forward()
      if variant=="force_controller":env.robots[0].controller.update(force=True)
     row["variants"].append(dict(variant=variant,pre_error=p.old.err(env.get_sim_state()[1:],st[t+1,1:]),
       controller_new_update=bool(env.robots[0].controller.new_update)))
     ob=p.step(env,act[t+1],d+"_"+variant,t+1)
     row["variants"][-1].update(post=v.ip.state_error(env,env.get_sim_state(),st[t+2]),
                                  observation=p.old.compare_obs(ob,g,t+1))
    v.write("RESULT.json",dict(status="PARTIAL",rows=rows))
    print(d,[(x["variant"],x["post"]["max_error"]) for x in row["variants"]],flush=True)
  for path,h in assets["input_hashes"].items():assert p.old.sha(path)==h
  v.write("RESULT.json",dict(status="ENGINEERING_DIAGNOSTIC_COMPLETE",rows=rows,inputs_unchanged=True,
                            policy_continuations=0,optimizer_calls=0))
 except BaseException as e:
  v.ledger("exception",error=repr(e));v.write("FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  if env:env.close();v.ledger("closed")
if __name__=="__main__":main()
