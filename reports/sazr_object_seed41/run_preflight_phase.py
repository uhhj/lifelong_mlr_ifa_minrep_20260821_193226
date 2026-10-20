"""Preserve an individual prerequisite's command, log and real exit code."""
import json,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
name=sys.argv[1]
commands={
 "software_cuda":["env","TEST_DEVICE=cuda",sys.executable,"-m","sazr_exp.tests"],
 "admission":[sys.executable,"-m","sazr_exp.admission"],
 "preflight":[sys.executable,"-m","sazr_exp.preflight"],
 "noise_interface":[sys.executable,"-m","sazr_exp.noise_check"],
 "freeze":[sys.executable,str(W/"freeze.py")],
}
assert name in commands and not (O/(name+".log")).exists()
with (O/"preflight_commands.jsonl").open("a") as f:f.write(json.dumps(dict(phase=name,command=commands[name],time=time.time()))+"\n")
with (O/(name+".log")).open("x") as log:r=subprocess.run(commands[name],stdout=log,stderr=subprocess.STDOUT)
(O/(name+".exit")).write_text(str(r.returncode)+"\n")
print((O/(name+".log")).read_text())
raise SystemExit(r.returncode)
