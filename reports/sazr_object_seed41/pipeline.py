import os,sys,time,json,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
def write(n,x):
    p=O/n;p.write_text(json.dumps(x,indent=2)+"\n")
assert not (O/"pipeline.started").exists()
write("pipeline.started",dict(pid=os.getpid(),time=time.time()))
rc=1
try:
    phases=[
        ("anchor",["-m","sazr_exp.rollout","anchor"]),
        ("anchors",["-m","sazr_exp.anchors"]),
        ("train",["-m","sazr_exp.train"]),
        ("validation_parent",["-m","sazr_exp.rollout","validation_parent"]),
        ("validation_candidate",["-m","sazr_exp.rollout","validation_candidate"]),
        ("decide",["-m","sazr_exp.decide"]),
        ("formal",["-m","sazr_exp.rollout","formal"]),
        ("audit",[str(W/"audit.py")])]
    for name,args in phases:
        assert not (O/(name+".log")).exists()
        with (O/"commands.jsonl").open("a") as f:f.write(json.dumps(dict(phase=name,command=[sys.executable]+args,time=time.time()))+"\n")
        with (O/(name+".log")).open("x") as f:
            tic=time.time();r=subprocess.run([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        (O/(name+".exit")).write_text(str(r.returncode)+"\n")
        with (O/"timings.jsonl").open("a") as f:f.write(json.dumps(dict(phase=name,seconds=time.time()-tic,exit=r.returncode))+"\n")
        assert r.returncode==0,(name,r.returncode)
    rc=0
except BaseException:
    write("BLOCKED.json",dict(status="BLOCKED",traceback=traceback.format_exc(),time=time.time()))
    write("RUN_STATUS.json",dict(status="BLOCKED",time=time.time()))
    raise
finally:(O/"pipeline.exit").write_text(str(rc)+"\n")
