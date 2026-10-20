"""Add finite LIBERO50 entries, with GPU and full input admission still mandatory."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu=json.loads((Q/"LIBERO50_CPU_NATIVE.json").read_text())
assert cpu["status"]=="CPU_NATIVE_PASS_NOT_GPU_ADMITTED"
assert cpu["optimizer_calls"]==cpu["environment_steps"]==cpu["cuda_allocated_bytes"]==0
assert cpu["history_lengths"]==[1,4,8]
assert (Q/"LIBERO50_FINAL_SOFTWARE.exit").read_text().strip()=="0"
bundle=Q/"LIBERO50_SOURCE_BUNDLE_final.json"
worker=Q/"worker_entrypoint_libero50.py"
shared_w=Q/"libero50/shared_pretraining"
shared_o=Path("/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021")
old=json.loads((Q/"REGISTRY.json").read_text());new=copy.deepcopy(old)
names=("PARTITION","PARENT_EXPORT_TEMPLATE","PARENT_TRAIN_TEMPLATE","PARENT_EVAL_TEMPLATE")
resources={str(Q/("LIBERO50_"+n+".json")):sha(Q/("LIBERO50_"+n+".json")) for n in names}
for seed in (41,42,43):
    key="RA_IFA_LIBERO50_%d"%seed
    assert key not in old
    w=Q/"libero50"/("seed%d"%seed)
    output=Path("/data/mlr_ifa/experiments/RA_Coreset_LIBERO50_seed%d_20261021"%seed)
    assert not w.exists() and not output.exists()
    entry=dict(method="RA_IFA",suite="LIBERO50",seed=seed,source=str(w/"source"),
        output=str(output),queue=str(Q),bundle=str(bundle),bundle_sha256=sha(bundle),
        shared_pretrain_workspace=str(shared_w),shared_pretrain_output=str(shared_o),
        resource_hashes=resources,implementation_reviewed=True,
        gpu_admitted=False,admission="per-worker real CUDA tests before any formal updates",
        source_disclosure="shared new 25-task pretraining reproduction; not an official released model")
    path=Q/"entries"/(key+".json");launcher=Q/"entries"/(key+".sh")
    assert not path.exists() and not launcher.exists()
    path.write_text(json.dumps(entry,indent=2)+"\n")
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\nsource /data/mlr_ifa/activate.sh\n"
                       "exec /data/mlr_ifa/env/bin/python %s %s\n"%(worker,path))
    deps=[worker,path,bundle,Q/"worker_entrypoint.py",Q/"queue_controller.py",Q/"queue_verifier.py",
          Q/"libero50_verifier.py",Q/"queue_controller_v2.py",Q/"branch_gate.py"]
    new[key]=dict(entry,launcher=str(launcher),launcher_sha256=sha(launcher),
        deployment_hashes={str(p):sha(p) for p in deps})
assert all(old[k]==new[k] for k in old)
stamp=str(time.time_ns())
(Q/("REGISTRY.before_LIBERO50."+stamp+".json")).write_bytes((Q/"REGISTRY.json").read_bytes())
tmp=Q/"REGISTRY.libero50.tmp";tmp.write_text(json.dumps(new,indent=2)+"\n");os.replace(tmp,Q/"REGISTRY.json")
(Q/"LIBERO50_REGISTRY_ACCEPTANCE.json").write_text(json.dumps(dict(status="REGISTERED_GPU_GATED",
    total_entries=len(new),added=sorted(set(new)-set(old)),preserved_entries=sorted(old),
    registry_sha256=sha(Q/"REGISTRY.json"),bundle_sha256=sha(bundle),unix=time.time(),
    formal_training_started=False,rollouts=0,CPU_probe_sha256=sha(Q/"LIBERO50_CPU_NATIVE.json")),indent=2)+"\n")
print("REGISTERED",len(new))
