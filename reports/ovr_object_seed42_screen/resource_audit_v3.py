"""Read-only OVR asset/initial-state provenance audit; no environment or model forward."""
import json,time,hashlib,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
P=Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019")
A=Path("/data/mlr_ifa/analyses/OVR_OBJECT_seed42_20261020_admission_v3")
A.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
init=Path("/data/mlr_ifa/src/libero/libero/init_files/libero_object/pick_up_the_butter_and_place_it_in_the_basket.pruned_init")
states=torch.load(init,map_location="cpu")
hashes=[hashlib.sha256(np.asarray(s).tobytes()).hexdigest() for s in states]
assert len(hashes)==len(set(hashes))==50
inputs={str(init):sha(init)}
assets={}
for stage in (7,9):
    cp=P/f"ra_coreset/models/mlrifa_{stage}_model_ep99.pth"
    inputs[str(cp)]=sha(cp);obj=torch.load(cp,map_location="cpu")
    sd={(k[7:] if k.startswith("module.") else k):v for k,v in obj["state_dict"].items()}
    keys={k:v for k,v in sd.items() if k.startswith(("temporal_transformer.","policy_head."))}
    assert len(keys)==100 and all(torch.isfinite(v).all() for v in keys.values())
    hh=hashlib.sha256()
    for k,v in sorted(keys.items()):
        hh.update(k.encode());hh.update(str((tuple(v.shape),v.dtype)).encode());hh.update(v.numpy().tobytes())
    assets[str(stage)]=dict(path=str(cp),sha256=inputs[str(cp)],bytes=cp.stat().st_size,
        downstream_tensor_keys=100,downstream_tensor_hash=hh.hexdigest(),
        shape_signature={k:list(v.shape) for k,v in keys.items()})
    assert sha(cp)==inputs[str(cp)]
assert assets["7"]["shape_signature"]==assets["9"]["shape_signature"]
historical={}
used=set()
for stage in (7,9):
    path=P/f"ra_coreset/evaluation/trials_stage{stage}_task6.jsonl"
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows)==20
    for r in rows:
        i=r["initial_state_index"];assert r["initial_state_sha256"]==hashes[i]
        used.add(i)
    inputs[str(path)]=sha(path)
    historical[str(stage)]=dict(source=str(path),sha256=sha(path),n=len(rows),
         successes=sum(r["success"] for r in rows),
         records=[dict(initial_index=r["initial_state_index"],initial_sha256=r["initial_state_sha256"],
                       success=r["success"]) for r in rows])
prior=[]
sazr=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
for phase in ("anchor","validation_parent","validation_candidate","formal"):
    path=sazr/phase/"episodes.jsonl";inputs[str(path)]=sha(path)
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids=[]
    for r in rows:
        i=r["initial_index"];assert r["initial_sha256"]==hashes[i];used.add(i);ids.append(i)
    prior.append(dict(source=str(path),sha256=sha(path),seed=41,phase=phase,
                      completed_records=len(rows),initial_indices=sorted(set(ids))))
# Cancelled independent evaluation: record committed and attempted use without resuming.
cancel=Path("/data/mlr_ifa/analyses/Coreset_validation_20261014/frozen_eval")
cancel_schema=[]
for folder in (cancel/"episodes",cancel/"smoke/episodes"):
    rr=[]
    for path in sorted(folder.glob("*.json")):
        row=json.loads(path.read_text());rr.append(row)
        inputs[str(path)]=sha(path)
    # Save metadata records only; no raw actions/features.
    safe=[]
    allow=("seed","arm","stage","task","initial_index","initial_state_index","init_index",
           "initial_state_sha256","initial_sha256","success","repeat_id","episode_id")
    for row in rr:
        safe.append({k:row[k] for k in allow if k in row})
    cancel_schema.append(dict(source=str(folder),n=len(rr),example_keys=sorted(rr[0]) if rr else [],
                              records=safe))
for path,h in inputs.items():assert sha(path)==h,path
report=dict(status="RESOURCE_AUDIT_COMPLETE_PROTOCOL_DECISION_PENDING",
    observed_at=time.time(),parent=P.as_posix(),checkpoints=assets,historical=historical,
    initial_file=str(init),initial_file_sha256=sha(init),available_states=50,
    unique_states=50,initial_hashes=hashes,confirmed_previously_used=sorted(used),
    not_used_by_reviewed_completed_ledgers=sorted(set(range(50))-used),
    usage_scope="Not a guarantee of never used anywhere; known SAZR and original RA ledgers, cancelled run separately listed.",
    prior_usage=prior,cancelled_evaluation=cancel_schema,input_hashes=inputs,
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"]).decode(),
    gpu_processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"]).decode(),
    data_disk_free_bytes=shutil.disk_usage("/data").free,system_disk_free_bytes=shutil.disk_usage("/root").free,
    new_model_forwards=0,optimizer_calls=0,environment_calls=0,
    needed_decisions=["independent vs exploratory state split and total budget",
                      "teacher superiority gate and insufficient success data handling",
                      "distillation loss lambda/gripper definition and scale handling",
                      "heldout distillation tolerance and enable gate"])
(A/"RESOURCE_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("status","historical","available_states",
    "confirmed_previously_used","gpu","gpu_processes","data_disk_free_bytes")},indent=2))
