"""Read-only snapshot including the versioned scheduler and server data acquisition."""
import json
import subprocess
from pathlib import Path
from monitor_snapshot import Q,read,run
def snapshot():
    result=run()
    result["controller_v2"]=dict(pid=read(Q/"controller_v2.pid.json"),
        handoff=read(Q/"CONTROLLER_V2_HANDOFF.json"),
        exit=(Q/"controller_v2.exit").read_text().strip() if (Q/"controller_v2.exit").exists() else None)
    assets=read(Q/"LIBERO50_SERVER_STATUS.json") or {}
    result["libero50_acquisition"]={k:v for k,v in assets.items() if k!="results"}
    result["libero50_acquisition"]["current"]=read(Q/"LIBERO50_SERVER_CURRENT.json")
    result["libero50_acquisition"]["exit"]=(Q/"libero50_acquisition.exit").read_text().strip() if (Q/"libero50_acquisition.exit").exists() else None
    shared=Path("/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021")
    result["libero50_shared_pretraining"]=read(shared/"RUN_STATUS.json")
    ps=subprocess.check_output(["ps","-eo","pid,ppid,etime,args"],text=True).splitlines()
    result["processes"]=[x for x in ps if any(m in x for m in
        ("rctr_exp.","caet_exp.","covr_exp.","queue_controller","l50_exp.","acquire_libero50_server"))]
    return result
if __name__=="__main__":print(json.dumps(snapshot(),indent=2))
