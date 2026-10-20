"""Bind a fresh full-training COVR run; old RA assets are read only."""
import json
import os
import shutil
import time
from pathlib import Path
import torch
from coreset_exp.common import O, SEED, write, sha, source_hashes


def main():
    parent=Path(os.environ["RA_BASE"])
    assert not O.exists()
    c=json.loads((parent/"resolved_ra_coreset.json").read_text())
    assert c["seed"]==SEED and c["benchmark_name"]=="LIBERO_GOAL"
    assert json.loads((parent/"PROCESS_EXIT_ACCEPTANCE.json").read_text())["status"]=="PASS"
    assert json.loads((parent/"AUDIT.json").read_text())["status"]=="PASS"
    tasks=json.loads((parent/"TASKS.json").read_text())
    for task in range(6,9):
        assert len(torch.load(tasks[str(task)]["paths"][2],map_location="cpu"))>=32
    O.mkdir(parents=True)
    files={}
    for name in ("FEATURES.json","MODAL_CONTRACT.json","PRETRAIN_METADATA.json","TASKS.json"):
        shutil.copy2(parent/name,O/name);files[str(parent/name)]=sha(parent/name)
    for folder in ("features","bindings","selection_preflight","offline_selection"):
        (O/folder).symlink_to((parent/folder).resolve(),target_is_directory=True)
    binding=dict(parent=str(parent),contract_sha256=sha(parent/"CONTRACT.json"),
        feature_manifest_sha256=sha(parent/"FEATURES.json"),modal_sha256=sha(parent/"MODAL_CONTRACT.json"),
        method="COVR_RA_IFA",seed=SEED)
    write(O/"PARENT_BINDING.json",binding)
    for name in ("ra_coreset","eval"):
        cfg=json.loads((parent/("resolved_%s.json"%name)).read_text())
        cfg.update(feature_dir=str(O/"features"),buffer_dir=str(O/"ra_coreset/buffer"),
            run_root=str(O/"ra_coreset"),experiment_dir=str(O/"ra_coreset/models"),
            contract=str(O/"CONTRACT.json"))
        write(O/("resolved_%s.json"%name),cfg)
    pc=json.loads((parent/"CONTRACT.json").read_text())
    files.update({p:h for p,h in pc.get("files",{}).items() if Path(p).is_file()})
    for p in (parent/"ra_coreset").rglob("*"):
        if p.is_file() and (p.suffix in (".json",".jsonl") or p.parent.name=="models"):
            files[str(p)]=sha(p)
    for p in O.iterdir():
        if p.is_file():files[str(p)]=sha(p)
    write(O/"CONTRACT.json",dict(version="COVR_GOAL_v1",seed=SEED,files=files,
        source_sha256=source_hashes(),modal_sha256=binding["modal_sha256"],
        epochs=400,optimizer_steps=76100,final_checkpoints=4,midpoint_checkpoints=3,
        formal_records=200,vector_attempts=20,normal_collection_episodes=36,
        exploration_branches=324,confirmation_branches=162,
        capture_steps_by_initial=[40,120,240,40,120,240],collection_initials=list(range(20,26)),
        diagnostic_initials=list(range(26,32)),
        diagnostic_rollouts=0,diagnostic_reason="Unbudgeted diagnostic initial set is reserved, not silently executed",
        candidate_priority="Max sum of FS versus Normal and argmax mean, then component index",
        training_continuation="First confirmation repetition rescuing both controls; at most8windows",
        loss=dict(coefficient=.1,execution_weight=.5,temperature=.25,auxiliary_batch=4,frequency=10),
        auxiliary_epoch_range=[50,99],sample_weights=[1.,.25],
        weight_denominator="Four-sample batch size",
        engineering_steps_per_selected_snapshot=16,preflight_optimizer_calls=4,
        preflight_environment_steps=28,created_unix=time.time()))
    write(O/"RUN_STATUS.json",dict(status="PREPARED_AWAITING_PREFLIGHT"))


if __name__=="__main__":
    main()
