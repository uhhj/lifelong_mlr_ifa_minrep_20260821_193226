"""Independent CAET run binding. Never reuse downstream trained policy parameters."""
import json
import os
import shutil
import time
from pathlib import Path
from coreset_exp.common import O, SEED, write, sha, source_hashes
from .variant import METHOD,VARIANT,ENTRY


def main():
    parent=Path(os.environ["RA_BASE"])
    assert not O.exists()
    c=json.loads((parent/"resolved_ra_coreset.json").read_text())
    assert c["seed"]==SEED and c["benchmark_name"]=="LIBERO_OBJECT"
    assert json.loads((parent/"AUDIT.json").read_text())["status"]=="PASS"
    assert json.loads((parent/"AUTONOMOUS_EXIT_ACCEPTANCE.json").read_text())["status"]=="PASS"
    O.mkdir(parents=True)
    files={str(ENTRY):sha(ENTRY)}
    for name in ("FEATURES.json","MODAL_CONTRACT.json","PRETRAIN_METADATA.json","TASKS.json"):
        shutil.copy2(parent/name,O/name)
        files[str(parent/name)]=sha(parent/name)
    for folder in ("features","bindings","selection_preflight","offline_selection"):
        (O/folder).symlink_to((parent/folder).resolve(),target_is_directory=True)
    binding=dict(parent=str(parent),contract_sha256=sha(parent/"CONTRACT.json"),
        feature_manifest_sha256=sha(parent/"FEATURES.json"),modal_sha256=sha(parent/"MODAL_CONTRACT.json"),
        method=METHOD,seed=SEED)
    write(O/"PARENT_BINDING.json",binding)
    for name in ("ra_coreset","eval"):
        cfg=json.loads((parent/("resolved_%s.json"%name)).read_text())
        cfg.update(feature_dir=str(O/"features"),buffer_dir=str(O/"ra_coreset/buffer"),
            run_root=str(O/"ra_coreset"),experiment_dir=str(O/"ra_coreset/models"),
            contract=str(O/"CONTRACT.json"))
        write(O/("resolved_%s.json"%name),cfg)
    pc=json.loads((parent/"CONTRACT.json").read_text())
    for path,h in pc.get("files",{}).items():
        if Path(path).is_file():
            files[path]=h
    for p in (parent/"ra_coreset").rglob("*"):
        if p.is_file() and (p.suffix in (".json",".jsonl") or p.parent.name=="models"):
            files[str(p)]=sha(p)
    for p in O.iterdir():
        if p.is_file():
            files[str(p)]=sha(p)
    for name in ('PROTOCOL.md','GATE1_EVIDENCE.json','GATE1_LOCAL_AUDIT.json'):
        shutil.copy2(Path(os.environ['CORESET_SOURCE']).parent/name,O/name)
        files[str(O/name)]=sha(O/name)
    write(O/"CONTRACT.json",dict(version=METHOD,variant=VARIANT,seed=SEED,
        files=files,source_sha256=source_hashes(),modal_sha256=binding["modal_sha256"],
        epochs=400,optimizer_steps=105500,final_checkpoints=4,
        formal_records=200,vector_attempts=20,auxiliary_coefficient=.05,
        effect_epochs=10,effect_batch=64,effect_lr=.001,effect_adamw_weight_decay=.01,
        holdout="Last numeric whole demo per task; main policy still uses all10",
        normalization="Train-only per-coordinate population std, floor1e-6",
        physical_target="Recorded target-object free-joint translation and local SO3 logarithm",
        auxiliary_head_lr=.0001,auxiliary_head_weight_decay=.01,auxiliary_head_supervision="next-action NLL on detached eight-step temporal output; no deployment-head gradient",auxiliary_preflight_optimizer_updates="see actual ledger",preflight_environment_steps=0,
        created_unix=time.time()))
    write(O/"RUN_STATUS.json",dict(status="PREPARED_AWAITING_PREFLIGHT"))


if __name__=="__main__":
    main()
