import os,sys,json,subprocess,time
from pathlib import Path
W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
assert not (O/"supervisor.started").exists()
(O/"supervisor.started").write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
with (O/"pipeline.log").open("x") as f:r=subprocess.run([sys.executable,str(W/"pipeline.py")],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
(O/"supervisor.exit").write_text(str(r.returncode)+"\n")
if r.returncode==0:
    assert (O/"pipeline.exit").read_text().strip()=="0"
    a=json.loads((O/"AUDIT.json").read_text());assert a["status"]=="PASS"
    (O/"PROCESS_EXIT_ACCEPTANCE.json").write_text(json.dumps(dict(status="PASS",pipeline_exit=0,supervisor_exit=0,time=time.time()),indent=2))
    (O/"RUN_STATUS.json").write_text(json.dumps(dict(status="COMPLETED",enabled=a["enabled"],time=time.time()),indent=2))
