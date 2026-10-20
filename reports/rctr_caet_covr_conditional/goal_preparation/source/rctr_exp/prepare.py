"""Bind an immutable same-seed RA run; copy metadata only and reuse feature bytes."""
import json
import os
import shutil
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from coreset_exp.common import O, SEED, S, write, save, sha, source_hashes


def main():
    parent = Path(os.environ["RA_BASE"])
    assert not O.exists(), "Never overwrite an existing run or frozen contract"
    parent_cfg = json.loads((parent/"resolved_ra_coreset.json").read_text())
    assert parent_cfg["seed"] == SEED
    assert parent_cfg["benchmark_name"] == "LIBERO_GOAL"
    assert (parent/"ra_coreset/TRAINING_COMPLETE.json").is_file()
    assert (parent/"ra_coreset/evaluation/EVALUATION_COMPLETE.json").is_file()
    assert json.loads((parent/"AUDIT.json").read_text())["status"] == "PASS"
    assert json.loads((parent/"PROCESS_EXIT_ACCEPTANCE.json").read_text())["status"] == "PASS"
    O.mkdir(parents=True)
    files = {}
    for name in ("FEATURES.json", "MODAL_CONTRACT.json", "PRETRAIN_METADATA.json", "TASKS.json"):
        shutil.copy2(parent/name, O/name)
        files[str(parent/name)] = sha(parent/name)
    (O/"features").symlink_to((parent/"features").resolve(), target_is_directory=True)
    for folder in ("bindings", "selection_preflight", "offline_selection"):
        if (parent/folder).exists():
            (O/folder).symlink_to((parent/folder).resolve(), target_is_directory=True)
    binding = dict(parent=str(parent), contract_sha256=sha(parent/"CONTRACT.json"),
                   feature_manifest_sha256=sha(parent/"FEATURES.json"),
                   modal_sha256=sha(parent/"MODAL_CONTRACT.json"),
                   method="RCTR_RA_IFA", seed=SEED)
    write(O/"PARENT_BINDING.json", binding)
    for name in ("ra_coreset", "eval"):
        c = json.loads((parent/("resolved_%s.json" % name)).read_text())
        c["feature_dir"] = str(O/"features")
        c["buffer_dir"] = str(O/"ra_coreset/buffer")
        c["run_root"] = str(O/"ra_coreset")
        c["experiment_dir"] = str(O/"ra_coreset/models")
        c["contract"] = str(O/"CONTRACT.json")
        write(O/("resolved_%s.json" % name), c)
    # Six pretraining tasks only. No future lifelong labels used for response scale.
    from libero.libero.benchmark import get_benchmark
    b = get_benchmark("LIBERO_GOAL")(0)
    moments = []
    for t in range(6):
        path = Path(parent_cfg["folder"])/b.get_task_demonstration(t)
        files[str(path)] = sha(path)
        with h5py.File(path) as f:
            for demo in sorted(f["data"], key=lambda s: int(s.split("_")[-1])):
                if demo.startswith("demo_"):
                    moments.append(f["data"][demo]["actions"][:].astype(np.float64))
    actions = np.concatenate(moments)
    scale = actions.std(0, ddof=0)
    assert np.isfinite(scale).all() and (scale[:6] > 1e-8).all()
    save(O/"pretraining_action_scale.pth", torch.tensor(scale, dtype=torch.float32))
    write(O/"ACTION_SCALE_METADATA.json", dict(source="Six original pretraining tasks",
        steps=len(actions), dimensions=7, ddof=0, scale_sha256=sha(O/"pretraining_action_scale.pth")))
    # Baseline metadata and all official data/initials remain protected.
    pc = json.loads((parent/"CONTRACT.json").read_text())
    for path, value in pc.get("files", {}).items():
        if Path(path).is_file() and Path(path).suffix in (".hdf5", ".pth", ".init", ".bddl"):
            files[path] = value
    for p in (parent/"ra_coreset").rglob("*"):
        if p.is_file() and (p.suffix in (".json", ".jsonl") or p.parent.name == "models"):
            files[str(p)] = sha(p)
    for p in O.iterdir():
        if p.is_file():
            files[str(p)] = sha(p)
    write(O/"CONTRACT.json", dict(version="RCTR_GOAL_v1", seed=SEED,
        files=files, source_sha256=source_hashes(), modal_sha256=binding["modal_sha256"],
        parent=binding, epochs=400, optimizer_steps=76100, final_checkpoints=4,
        formal_records=200, vector_attempts=20, exploration_branch_limit=1368,
        confirmation_branch_limit=144, collection_initials=list(range(20, 26)),
        capture_steps=[40, 120], coefficients=dict(response=.1, route=.02, preference=.02),
        control_scale="population std over six pretraining task expert actions; first6 only",
        nearest_component="squared distance, first6 standardized by pretraining scale; gripper raw",
        preference="negative mean squared distance under actual argmax Gaussian std, all7",
        record_order="task then lexical initial_snapshot; global optimizer step modulo record count",
        confirmation_selection="first exploration-qualified axis/sign/nonwinner rank; no replacement",
        preflight_engineering_steps=28, preflight_optimizer_updates=0,
        prefix_policy_steps_per_task=720, prefix_zero_steps_per_task=30,
        engineering_repeat_per_available_snapshot=16, native_tolerance=1e-5,
        state_observation_tolerance=1e-10, created_unix=time.time()))
    write(O/"RUN_STATUS.json", dict(status="PREPARED_AWAITING_PREFLIGHT", unix=time.time()))


if __name__ == "__main__":
    main()
