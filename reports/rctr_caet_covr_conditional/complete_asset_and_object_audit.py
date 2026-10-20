"""Read-only full data/feasibility acceptance; zero policy/environment execution."""
import json
import subprocess
import sys
import time
from pathlib import Path
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
text=(Q/"OBJECT_EFFECT_FEASIBILITY_attempt2.log").read_text()
decoder=json.JSONDecoder();report=None
for i,c in enumerate(text):
    if c!="{":continue
    try:value,_=decoder.raw_decode(text[i:])
    except ValueError:continue
    if isinstance(value,dict) and value.get("kind")=="READ_ONLY_FEASIBILITY_NOT_FITTED_OR_GPU":
        report=value;break
assert report and report["status"]=="PASS"
assert all(r["pair_feasible"] for r in report["seeds"].values())
(Q/"OBJECT_EFFECT_FEASIBILITY.json").write_text(json.dumps(report,indent=2)+"\n")
acquisition=json.loads((Q/"LIBERO50_SERVER_STATUS.json").read_text())
if acquisition["status"]!="DATA_VERIFIED" or not (Q/"libero50_acquisition.exit").exists():
    print(json.dumps(dict(status="DATA_STILL_ACQUIRING",verified=acquisition["verified"],
        OBJECT_CAET_pair_feasible=True)));sys.exit(0)
assert acquisition["verified"]==50 and (Q/"libero50_acquisition.exit").read_text().strip()=="0"
previous=Q/"LIBERO50_ASSET_PROBE.json"
backup=Q/"LIBERO50_ASSET_PROBE_before_complete.json"
if previous.exists() and not backup.exists():backup.write_bytes(previous.read_bytes())
with (Q/"LIBERO50_FULL_ASSET_AUDIT.log").open("x") as f:
    rc=subprocess.run([sys.executable,str(Q/"probe_libero50_assets.py")],
        stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL).returncode
(Q/"LIBERO50_FULL_ASSET_AUDIT.exit").write_text(str(rc)+"\n")
assert rc==0
assets=json.loads(previous.read_text())
assert assets["status"]=="DATA_AND_ENV_METADATA_READY" and assets["verified"]==50
result=dict(status="DATA_AND_CPU_READINESS_PASS_NOT_GPU_ADMITTED",unix=time.time(),
    acquisition_exit=0,official_data_files=50,all_packed_files_rehashed=True,
    actual_main_steps_per_seed=assets["main_steps_per_seed"],
    actual_shared_pretrain_steps=assets["shared_pretrain_steps"],
    stage_counts=assets["stage_counts"],pretrain_checkpoint_available=False,
    object_CAET_positive_pairs={s:{t:v["positive_pairs"] for t,v in r["stages"].items()}
                               for s,r in report["seeds"].items()},
    optimizer_updates=0,environment_steps=0)
(Q/"READINESS_20261021.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
