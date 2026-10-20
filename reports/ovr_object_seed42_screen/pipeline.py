"""Finite foreground controller; every child is a separate process."""
import os,sys,time,json,subprocess
from pathlib import Path
W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/experiments/OVR_OBJECT_RA_seed42_screen_20261020")
def main():
    assert not (W/"pipeline.started").exists()
    (W/"pipeline.started").write_text(str(time.time()))
    (W/"pipeline.pid").write_text(str(os.getpid()))
    rc=1
    try:
        for label,args in [("parent9",["rollout","9"]),("teacher7",["rollout","7"]),("audit",["audit"])]:
            cmd=[sys.executable,str(W/"screen.py")]+args
            with (W/"commands.jsonl").open("a") as f:f.write(json.dumps(dict(start=time.time(),label=label,cmd=cmd))+"\n")
            with (W/(label+".log")).open("x") as f:
                r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            (W/(label+".exit")).write_text(str(r.returncode))
            with (W/"commands.jsonl").open("a") as f:f.write(json.dumps(dict(end=time.time(),label=label,exit=r.returncode))+"\n")
            if r.returncode:raise RuntimeError(label+" failed; no automatic retry")
        rc=0
    finally:(W/"pipeline.exit").write_text(str(rc))
if __name__=="__main__":main()
