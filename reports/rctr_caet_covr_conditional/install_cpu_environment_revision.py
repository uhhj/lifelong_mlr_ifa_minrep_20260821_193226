"""Version only not-started launchers; never change active source or contracts."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((Q/"REGISTRY.json").read_text());new=copy.deepcopy(old);changed=[]
assert "CPU_IMPORT_PASS CUDA_ALLOCATED 0" in (Q/"CPU_NATIVE_IMPORT_retry.log").read_text()
worker=Q/"worker_entrypoint_env_v2.py"
assert worker.is_file()
for key,entry in old.items():
    if (Q/"starts"/(key+".json")).exists() or Path(entry["output"]).exists():
        continue
    if entry["method"] not in ("RCTR","CAET","COVR"):continue
    original=Q/"entries"/(key+".json")
    value=json.loads(original.read_text())
    revised=Q/"entries"/(key+".env_v2.json")
    launcher=Q/"entries"/(key+".env_v2.sh")
    assert not revised.exists() and not launcher.exists()
    value["launch_environment_revision"]="CPU tensors with valid CUDA/EGL mapping; zero CUDA bytes import probe"
    revised.write_text(json.dumps(value,indent=2)+"\n")
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\nsource /data/mlr_ifa/activate.sh\n"
        "exec /data/mlr_ifa/env/bin/python %s %s\n"%(worker,revised))
    hashes={p:h for p,h in entry["deployment_hashes"].items()
            if Path(p).name not in ("worker_entrypoint.py","worker_entrypoint_goal.py",key+".json")}
    hashes[str(worker)]=sha(worker)
    # Import-only install() dependency remains byte-frozen.
    hashes[str(Q/"worker_entrypoint.py")]=sha(Q/"worker_entrypoint.py")
    hashes[str(revised)]=sha(revised)
    new[key].update(launcher=str(launcher),launcher_sha256=sha(launcher),deployment_hashes=hashes)
    changed.append(key)
assert changed and new["RCTR_OBJECT_42"]==old["RCTR_OBJECT_42"]
for key in changed:
    for p,h in new[key]["deployment_hashes"].items():assert sha(p)==h,p
stamp=str(time.time_ns())
(Q/("REGISTRY.before_CPU_environment."+stamp+".json")).write_bytes((Q/"REGISTRY.json").read_bytes())
tmp=Q/"REGISTRY.env.tmp";tmp.write_text(json.dumps(new,indent=2)+"\n");os.replace(tmp,Q/"REGISTRY.json")
(Q/"CPU_ENVIRONMENT_REVISION.json").write_text(json.dumps(dict(status="PASS",changed=changed,
    preserved=[k for k in old if k not in changed],unix=time.time(),registry_sha256=sha(Q/"REGISTRY.json"),
    reason="native metric import derives EGL device ID from CUDA_VISIBLE_DEVICES; empty value crashes",
    scientific_source_changes=0,training_updates=0,environment_steps=0,
    import_probe_cuda_bytes=0,old_launchers_and_failures_preserved=True),indent=2)+"\n")
print("REVISED_UNSTARTED",len(changed))
