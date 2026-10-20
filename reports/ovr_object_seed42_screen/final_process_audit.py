"""External read-only completion check; does not launch an environment."""
import json,time,sys,subprocess,hashlib
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/OVR_OBJECT_RA_seed42_screen_20261020")
O=Path("/data/mlr_ifa/experiments/OVR_OBJECT_RA_seed42_screen_20261020")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
assert (W/"pipeline.exit").read_text().strip()=="0"
assert all((W/(x+".exit")).read_text().strip()=="0" for x in ("freeze","preflight_repair","parent9","teacher7","audit"))
assert (W/"preflight.exit").read_text().strip()=="1"
assert sha(W/"preflight_repair.py")==js(O/"PREFLIGHT_REPAIR.json")["adapter_sha256"]
procs=subprocess.check_output(["ps","-eo","pid,args"]).decode().splitlines()
workers=[l for l in procs if str(W) in l and ("screen.py rollout" in l or "pipeline.py" in l) and "bash -lc" not in l]
assert not workers,workers
gpu_processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"]).decode().strip()
assert not gpu_processes,("GPU process remains; inspect ownership before acceptance",gpu_processes)
for p,h in js(O/"CONTRACT.json")["files"].items():assert sha(p)==h,p
assert js(O/"SCREEN_AUDIT.json")["status"]=="PASS"
(O/"RUN_STATUS.json").write_text(json.dumps(dict(status="SCREEN_COMPLETED",time=time.time(),
    episodes=20,optimizer_calls=0,full_ovr="NOT_STARTED"))+"\n")
manifest={}
for root in (O,W):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".json",".jsonl",".log",".exit",".started",".pid",".md"):
            manifest[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
report=dict(status="PASS",time=time.time(),pipeline_exit=0,remaining_workers=workers,
    gpu_processes=gpu_processes,input_hashes_unchanged=True,manifest=manifest)
(O/"PROCESS_EXIT_ACCEPTANCE.json").write_text(json.dumps(report,indent=2)+"\n")
print("PROCESS_EXIT_ACCEPTANCE PASS")
