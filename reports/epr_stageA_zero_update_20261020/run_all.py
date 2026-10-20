"""Finite inference-only two-suite runner; no restart or automatic next stage."""
from pathlib import Path
import subprocess,json,shlex,time
W=Path(__file__).resolve().parent
A=Path("/data/mlr_ifa/analyses/EPR_stageA_20261020")
assert not (W/"RUN_STARTED.json").exists(),"Already started; inspect failure instead"
write=lambda p,x:p.write_text(json.dumps(x,indent=2)+"\n")
write(W/"RUN_STARTED.json",dict(time=time.time(),pid=__import__("os").getpid()))
jobs=[
 ("GOAL42","/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh"),
 ("OBJECT41","/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed41/env.sh")]
for name,env in jobs:
    cmd="source "+shlex.quote(env)+" && python "+shlex.quote(str(W/"run_diagnostic.py"))+" --suite "+name+" --out "+shlex.quote(str(A/name))
    write(W/(name+"_COMMAND.json"),dict(command=cmd,time=time.time()))
    with (W/(name+".log")).open("xb") as f:r=subprocess.run(["bash","-lc",cmd],stdout=f,stderr=subprocess.STDOUT)
    (W/(name+".exit")).write_text(str(r.returncode)+"\n")
    if r.returncode:
        write(W/"RUN_EXIT.json",dict(exit=r.returncode,failed_suite=name,time=time.time()))
        raise SystemExit(r.returncode)
write(W/"RUN_EXIT.json",dict(exit=0,time=time.time()))
