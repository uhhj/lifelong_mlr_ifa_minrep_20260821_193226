"""Recorded HDF5 transitions only: no simulator restoration or action integration."""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from coreset_exp.common import O, verify, sha, save, write

TARGETS = {6:"butter", 7:"milk", 8:"chocolate_pudding", 9:"orange_juice"}
DIMENSIONS = {"hinge":(1,1), "slide":(1,1), "ball":(4,3), "free":(7,6)}


def joint_layout(xml, target):
    tree = ET.fromstring(xml)
    q, v, found, names = 0, 0, None, []
    # MuJoCo flat state: time, qpos in kinematic-tree joint order, qvel, activation.
    # Traverse only actual worldbody joints; default-class templates are not joints.
    for node in tree.find("worldbody").iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        dq, dv = DIMENSIONS[kind]
        name = node.get("name")
        names.append(name)
        if name == target+"_1_joint0":
            assert kind == "free" and found is None
            found = q
        q += dq
        v += dv
    assert names[:7] == ["robot0_joint%d" % i for i in range(1,8)]
    assert names[7:9] == ["gripper0_finger_joint1", "gripper0_finger_joint2"]
    assert found is not None
    return found, q, v, names


def recorded_effects(group, target):
    xml = group.attrs["model_file"]
    if isinstance(xml, bytes):
        xml = xml.decode()
    offset, nq, nv, names = joint_layout(xml, target)
    state = group["states"][:].astype(np.float64)
    actions = group["actions"][:]
    joints = group["obs/joint_states"][:]
    grip = group["obs/gripper_states"][:]
    assert state.shape[0] == len(actions) == len(joints) == len(grip)
    assert state.shape[1] == 1+nq+nv, "Unsupported activation/state layout"
    # Independent actual-file validation of the converter's post-action observation.
    assert np.array_equal(state[1:,1:8], joints[:-1])
    assert np.array_equal(state[1:,8:10], grip[:-1])
    pose = state[:,1+offset:1+offset+7]
    assert np.allclose(np.linalg.norm(pose[:,3:],axis=1), 1, atol=1e-6, rtol=0)
    quat_xyzw = pose[:,[4,5,6,3]]
    rotation = Rotation.from_quat(quat_xyzw)
    effect = np.concatenate((np.diff(pose[:,:3],axis=0),
        (rotation[:-1].inv()*rotation[1:]).as_rotvec()),axis=1)
    assert np.isfinite(effect).all()
    return effect, dict(qpos_offset=offset, nq=nq, nv=nv, joint_names=names,
        joint_and_gripper_alignment_exact=True,
        convention="action[t]: states[t] -> states[t+1]; obs[t] is post-action",
        xml_sha256=__import__("hashlib").sha256(xml.encode()).hexdigest())


def build(task, root=None):
    verify()
    root = root or O/"effects/current"/("task%d" % task)
    assert not (root/"labels.pth").exists()
    binding = json.loads((O/"bindings"/("task%d.json" % task)).read_text())
    assert sha(binding["hdf5"]) == binding["hdf5_sha256"]
    selected_demos = sorted({x["demo_id"] for x in binding["windows"].values()
                             if x["identity_unique"]}, key=lambda s:int(s.split("_")[-1]))
    holdout = selected_demos[-1]  # Whole trajectory; never random overlapping windows.
    values, demo_audits, counts = {}, {}, Counter()
    with h5py.File(binding["hdf5"], "r") as h:
        for demo in selected_demos:
            effect, demo_audits[demo] = recorded_effects(h["data"][demo], TARGETS[task])
            values[demo] = effect
        labels = {}
        for sample_id, b in binding["windows"].items():
            if not b["identity_unique"]:
                reason = "AMBIGUOUS_IDENTITY"
            elif b["target"] >= len(values[b["demo_id"]]):
                reason = "NO_RECORDED_NEXT_STATE"
            else:
                reason = None
            valid = reason is None
            effect = values[b["demo_id"]][b["target"]].astype(np.float32) if valid else None
            moving = valid and (np.linalg.norm(effect[:3]) >= .001 or
                                np.linalg.norm(effect[3:]) >= np.pi/180)
            labels[sample_id] = dict(valid=valid, invalid_reason=reason,
                effect=torch.from_numpy(effect) if valid else None, moving=bool(moving),
                sample_id=sample_id, sample_sha256=b["sha256"], task=task,
                demo_id=b["demo_id"], target_step=b["target"], holdout=b["demo_id"]==holdout)
            counts["valid" if valid else reason] += 1
            counts["moving"] += bool(moving)
            counts["holdout_valid"] += valid and b["demo_id"] == holdout
    assert counts["valid"] > 0 and counts["moving"] > 0 and counts["holdout_valid"] > 0
    save(root/"labels.pth", labels)
    write(root/"AUDIT.json", dict(status="PASS", task=task, target_object=TARGETS[task],
        count=len(labels), counts=counts, holdout_demo=holdout, demos=demo_audits,
        source_hdf5_sha256=binding["hdf5_sha256"], labels_sha256=sha(root/"labels.pth"),
        environment_steps=0, optimizer_calls=0,
        boundary="Preserves existing post-action observation/action pairing of original dataset; "
                  "effect uses recorded physical before/after states, not predicted causality."))
    return labels


def retain(task):
    current = O/"effects/current"/("task%d" % task)
    labels = torch.load(current/"labels.pth", map_location="cpu")
    manifest = json.loads((O/"ra_coreset/storage"/("task%d.json" % task)).read_text())
    kept = {}
    for item in manifest["selected"]:
        original = Path(item["source"]).name
        assert labels[original]["sample_sha256"] == item["sha256"]
        kept[item["destination"]] = labels[original]
    dest = O/"effects/memory"/("task%d.pth" % task)
    if dest.exists():
        previous = torch.load(dest, map_location="cpu")
        assert set(previous) == set(kept)
        for k in kept:
            assert previous[k]["sample_sha256"] == kept[k]["sample_sha256"]
        return
    save(dest, kept)
    write(O/"effects/memory"/("task%d.json" % task),
          dict(status="PASS", retained=len(kept), sha256=sha(dest),
               storage_manifest_sha256=sha(O/"ra_coreset/storage"/("task%d.json" % task))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=range(6,10))
    ap.add_argument("--retain", action="store_true")
    args = ap.parse_args()
    retain(args.task) if args.retain else build(args.task)


if __name__ == "__main__":
    main()
