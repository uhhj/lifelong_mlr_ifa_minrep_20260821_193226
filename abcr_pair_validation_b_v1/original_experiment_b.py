#!/usr/bin/env python3
"""ABCR Experiment B: CPU-only, fail-closed physical/visual pair validation.

Commands:
  inspect-h5  --h5 DATASET.hdf5
  calibrate   --records BASE.npz --pairs BASE_LABELED.csv --encoder-sha256 SHA --out frozen.json
  screen      --records ALL.npz  --pairs CANDIDATES.csv --frozen frozen.json --out-dir report

This program DOES NOT reconstruct mappings from .pt record numbers. 'alignment_verified'
means the operator has independently verified each .pt/HDF5 window alignment.
No model training, CUDA, or original RGB images are used here.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SCHEMA = "abcr_experiment_b_v1"
DEFAULTS = {"position_m": 0.03, "orientation_deg": 15.0,
            "gripper_open_fraction": 0.10, "tau_s": 1.0,
            "linear_velocity_mps": 0.05, "angular_velocity_radps": 0.35,
            "fpr_quantile": 0.95, "motion_share": 0.5,
            "motion_bif": 1.5}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, data, overwrite=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Frozen protocol already exists: {p}; never overwrite a calibrated threshold")
    temp = p.with_name(p.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_records(path):
    """All entries must share the stable ID used by calibration/candidate CSVs.

    Required NPZ arrays, first dimension N:
      ids[N]                        stable .pt-derived record IDs / paths (strings)
      task_ids[N]                   task IDs (integers)
      traj_ids[N]                   independently verified demo IDs (strings)
      alignment_verified[N]        BOOLEAN, verified .pt <-> HDF5 window matches
      eef_pos_m[N,3]                world/base-frame Cartesian position in metres
      eef_quat_xyzw[N,4]            world/base-frame quaternion (x,y,z,w)
      gripper_open_fraction[N]      normalized physical opening, 0=closed, 1=open
      agent_unmod[N,D1]             image encoder feature BEFORE language FiLM
      eih_unmod[N,D2]               image encoder feature BEFORE language FiLM
      future_actions[N,Q,7]         current time through Q-1 future expert commands
      future_valid[N,Q]             valid, same-trajectory future action mask

    Optional, but STRONGLY preferred for dynamical-state comparability:
      eef_linvel_mps[N,3]          end-effector linear velocity in base frame
      eef_angvel_radps[N,3]        end-effector angular velocity in base frame
    Both velocity arrays must be jointly present or jointly absent.

    Never treat .pt stem as a timestep. Use verified mapping before export.
    """
    fields = ["ids", "task_ids", "traj_ids", "alignment_verified", "eef_pos_m",
              "eef_quat_xyzw", "gripper_open_fraction", "agent_unmod",
              "eih_unmod", "future_actions", "future_valid"]
    with np.load(path, allow_pickle=False) as z:
        missing = set(fields) - set(z.files)
        if missing:
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        d = {x: z[x] for x in fields}
        vel_keys = ("eef_linvel_mps", "eef_angvel_radps")
        available = [k in z.files for k in vel_keys]
        if any(available) and not all(available):
            raise ValueError("Both linear and angular velocity fields must be jointly supplied")
        if all(available):
            d.update({x:z[x] for x in vel_keys})
    ids = [str(x) for x in d["ids"].tolist()]
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError("Duplicate record IDs: cannot safely map candidates")
    for k, v in d.items():
        if len(v) != n:
            raise ValueError(f"{k} has {len(v)} records, expected {n}")
    if d["eef_pos_m"].shape != (n, 3) or d["eef_quat_xyzw"].shape != (n, 4):
        raise ValueError("eef_pos_m must be Nx3, eef_quat_xyzw Nx4")
    for vk in ("eef_linvel_mps", "eef_angvel_radps"):
        if vk in d and d[vk].shape != (n,3):
            raise ValueError(f"{vk} must be Nx3")
    if d["gripper_open_fraction"].shape != (n,):
        raise ValueError("gripper_open_fraction must be shape N")
    for key in ("agent_unmod", "eih_unmod"):
        if d[key].ndim != 2 or d[key].shape[1] == 0:
            raise ValueError(f"{key} must be NxD")
    a, valid = d["future_actions"], d["future_valid"]
    if a.ndim != 3 or a.shape[-1] != 7 or valid.shape != a.shape[:2]:
        raise ValueError("future_actions must be NxQx7 and future_valid NxQ")
    for k in ("eef_pos_m", "eef_quat_xyzw", "gripper_open_fraction",
              "agent_unmod", "eih_unmod", "future_actions"):
        if not np.isfinite(d[k]).all():
            raise ValueError(f"Non-finite data in {k}")
    if not np.all(np.isin(d["alignment_verified"], (True, False))):
        raise ValueError("alignment_verified must contain boolean flags")
    if not np.all(np.isin(valid, (True, False))):
        raise ValueError("future_valid must contain boolean flags")
    if (d["gripper_open_fraction"].min() < -1e-6 or
            d["gripper_open_fraction"].max() > 1+1e-6):
        raise ValueError("gripper_open_fraction must represent physical opening normalized to [0,1]")
    quat_norm = np.linalg.norm(d["eef_quat_xyzw"], axis=1)
    if np.any(quat_norm < 1e-6) or np.any(abs(quat_norm - 1) > 0.05):
        raise ValueError("Invalid orientation quaternion: ensure correct quaternion order and export")
    for k in ("agent_unmod", "eih_unmod"):
        if np.any(np.linalg.norm(d[k], axis=1) <= 1e-8):
            raise ValueError(f"Zero-norm {k} feature")
    d["ids"] = ids
    d["index"] = {rid:i for i,rid in enumerate(ids)}
    return d


def get_pairs(path, d, require_labels=False):
    """Read pairs id_a,id_b[,label]; also accepts original A new_file,old_file."""
    pairs = read_csv(path)
    if not pairs:
        raise ValueError("No candidate/calibration rows")
    out, seen = [], {}
    for line, row in enumerate(pairs, start=2):
        ida = row.get("id_a") or row.get("new_id") or row.get("new_file")
        idb = row.get("id_b") or row.get("old_id") or row.get("old_file")
        if not ida or not idb:
            raise ValueError(f"Pair CSV line {line}: requires id_a,id_b or new_file,old_file")
        if ida not in d["index"] or idb not in d["index"]:
            raise ValueError(f"Line {line}: missing mapped record: {ida!r}, {idb!r}")
        i,j = d["index"][ida], d["index"][idb]
        if i == j:
            raise ValueError(f"Line {line}: same record on both sides")
        if not (bool(d["alignment_verified"][i]) and bool(d["alignment_verified"][j])):
            raise ValueError(f"Line {line}: HDF5 window alignment is NOT independently verified")
        if require_labels:
            label = str(row.get("label", "")).strip().lower()
            if label not in ("0", "1", "match", "nonmatch"):
                raise ValueError(f"Line {line}: label must be 0/1 or nonmatch/match")
            label = int(label in ("1", "match"))
        else:
            label = None
        key = (min(i,j), max(i,j))
        if key in seen:
            # Duplicates from multiple A thresholds are okay only when labels agree.
            # Never silently discard contradictory manual labels.
            if seen[key] != label:
                raise ValueError(f"Line {line}: contradictory labels for repeated pair")
            continue
        seen[key] = label
        out.append((i,j,label))
    return out


def pair_descriptors(d, pairs, pos_m, orientation_deg, grip_frac,
                     use_velocity=False, velocity_tol=0.05, angvelocity_tol=0.35):
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    pos = np.linalg.norm(d["eef_pos_m"][ii] - d["eef_pos_m"][jj], axis=1)
    qa = d["eef_quat_xyzw"][ii].astype(np.float64)
    qb = d["eef_quat_xyzw"][jj].astype(np.float64)
    qa /= np.linalg.norm(qa, axis=1, keepdims=True)
    qb /= np.linalg.norm(qb, axis=1, keepdims=True)
    qcos = np.clip(np.abs(np.sum(qa*qb, axis=1)), 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(qcos)
    grip = abs(d["gripper_open_fraction"][ii] - d["gripper_open_fraction"][jj])
    ds_terms = [pos/pos_m, angle_rad/math.radians(orientation_deg), grip/grip_frac]
    dv = None
    dw = None
    if use_velocity:
        if "eef_linvel_mps" not in d or "eef_angvel_radps" not in d:
            raise ValueError("Calibration requires velocity matching but records lack both velocity arrays")
        dv = np.linalg.norm(d["eef_linvel_mps"][ii]-d["eef_linvel_mps"][jj], axis=1)
        dw = np.linalg.norm(d["eef_angvel_radps"][ii]-d["eef_angvel_radps"][jj], axis=1)
        ds_terms.extend([dv/velocity_tol, dw/angvelocity_tol])
    ds = np.maximum.reduce(ds_terms)

    def cosine(key):
        aa = d[key][ii].astype(np.float64)
        bb = d[key][jj].astype(np.float64)
        return np.einsum('nd,nd->n', aa, bb) / (np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1))
    agent_cos = cosine("agent_unmod")
    eih_cos = cosine("eih_unmod")
    # Conservative: neither view is allowed to mask a low similarity in the other.
    sv = np.minimum(agent_cos, eih_cos)
    return {"ds":ds, "sv":sv, "pos_m":pos, "rot_deg":np.rad2deg(angle_rad),
            "grip_delta":grip, "agent_cos":agent_cos, "eih_cos":eih_cos,
            "velocity_delta_mps":dv, "angular_velocity_delta_radps":dw}


def frozen_action_scale(d, max_task_id):
    base = d["task_ids"].astype(np.int64) <= max_task_id
    a, valid = d["future_actions"][base], d["future_valid"][base].astype(bool)
    mot = a[valid][:,:6]
    if len(mot) < 40:
        raise ValueError("Insufficient base-task future expert actions to freeze a 6D motion scale")
    return np.maximum(np.quantile(mot, .75, axis=0)-np.quantile(mot, .25, axis=0), .1)


def cmd_calibrate(args):
    d = load_records(args.records)
    pairs = get_pairs(args.pairs, d, require_labels=True)
    if any(int(d["task_ids"][i]) > args.max_base_task or int(d["task_ids"][j]) > args.max_base_task
           for i,j,_ in pairs):
        raise ValueError("Calibration must use only preregistered BASE tasks; incremental-task labels are forbidden")
    if any(int(d["task_ids"][i]) == int(d["task_ids"][j]) for i,j,_ in pairs):
        raise ValueError("Calibration pairs must be cross-task; do not use same-task easy positives")
    if min(args.pos_m, args.rot_deg, args.grip_frac, args.velocity_mps, args.angular_velocity_radps) <= 0:
        raise ValueError("Physical-state tolerances must be positive")
    if d["future_actions"].shape[1] < args.q:
        raise ValueError("Calibration future action arrays are shorter than --q")
    use_vel="eef_linvel_mps" in d
    desc = pair_descriptors(d,pairs,args.pos_m,args.rot_deg,args.grip_frac,
                            use_velocity=use_vel,velocity_tol=args.velocity_mps,
                            angvelocity_tol=args.angular_velocity_radps)
    physically = desc["ds"] <= 1.0
    labels = np.array([p[2] for p in pairs], dtype=bool)
    pos_scores = desc["sv"][physically & labels]
    neg_scores = desc["sv"][physically & ~labels]
    if len(pos_scores) < args.min_pos or len(neg_scores) < args.min_neg:
        raise ValueError(f"Not enough independently annotated, physically admissible pairs: "
                         f"positive {len(pos_scores)} (need {args.min_pos}), "
                         f"negative {len(neg_scores)} (need {args.min_neg}). "
                         "Do not substitute pseudo-labels from cosine thresholds.")
    # Freeze the threshold to a hard-negative quantile on BASE tasks only.
    try:
        qv = np.quantile(neg_scores, args.negative_quantile, method="higher")
    except TypeError:  # numpy < 1.22
        qv = np.quantile(neg_scores, args.negative_quantile, interpolation="higher")
    tau_v = min(1.0, float(qv) + 1e-7)
    positive_recall = float((pos_scores >= tau_v).mean())
    negative_fpr = float((neg_scores >= tau_v).mean())
    motion_scale = frozen_action_scale(d,args.max_base_task)
    sha = args.encoder_sha256.lower()
    if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError("--encoder-sha256 needs 64 hex digits for the exact frozen pre-FiLM checkpoint")
    record = {
        "schema":SCHEMA,"protocol_status":"FROZEN_BASE_ONLY_NO_INCREMENTAL_RETUNING",
        "physical": {"distance": "max(position_error/pos_tol, rotation_error/rot_tol, gripper_error/grip_tol)",
                     "tau_s":1.0,"pos_tolerance_m":args.pos_m,"rot_tolerance_deg":args.rot_deg,
                     "grip_tolerance_fraction":args.grip_frac,"quaternion_order":"xyzw",
                     "frame":"same robot base/world frame",
                     "use_velocity":use_vel,
                     "velocity_tolerance_mps":args.velocity_mps,
                     "angular_velocity_tolerance_radps":args.angular_velocity_radps,
                     "velocity_limitations":"POSE_ONLY_NOT_FULL_DYNAMIC_STATE" if not use_vel else "VELOCITY_INCLUDED"},
        "visual": {"representation":"two independently normalized, pre-FiLM frozen image embeddings",
                   "similarity":"min(cos(agent),cos(eih))","tau_v":tau_v,
                   "selection":"95th percentile by default of independently labeled hard negatives with ds<=1, BASE tasks only",
                   "negative_quantile":args.negative_quantile},
        "action": {"q":args.q,"share_max_motion_distance":args.motion_share,
                   "bif_min_motion_distance":args.motion_bif,
                   "motion_scale_base_IQR_floor_0p1":motion_scale.tolist(),
                   "gripper_conflict":"any difference in sign of gripper expert command within valid q steps"},
        "calibration": {"max_base_task":args.max_base_task,
                        "n_pairs_labeled":len(pairs),"n_physical_positive":len(pos_scores),
                        "n_physical_negative":len(neg_scores),
                        "positive_recall_at_tau_v":positive_recall,
                        "negative_fpr_at_tau_v":negative_fpr,
                        "n_distinct_trajectory_pairs":len(set((str(d['traj_ids'][i]),str(d['traj_ids'][j])) for i,j,_ in pairs)),
                        "records_sha256":sha256(args.records),"labels_csv_sha256":sha256(args.pairs),
                        "encoder_checkpoint_sha256":sha},
        "limitations":"Heuristic physical tolerances; a frozen calibration threshold is not ground-truth correctness. "
                      "The score is conditional on physically gated candidates, and calibrated rates do not "
                      "constitute independent-trajectory confidence intervals."
    }
    dump_json(args.out,record)
    print(json.dumps({"frozen_path":args.out,"tau_s":1,"tau_v":tau_v,
                      "base_positive_recall":positive_recall,"base_negative_FPR":negative_fpr,
                      "n_pos":len(pos_scores),"n_neg":len(neg_scores)},indent=2))


def motion_pair(d, pairs, q, scale):
    ii = np.array([p[0] for p in pairs],dtype=int)
    jj = np.array([p[1] for p in pairs],dtype=int)
    a = d["future_actions"]
    v = d["future_valid"].astype(bool)
    if a.shape[1] < q:
        raise ValueError(f"future_actions only has Q={a.shape[1]}, frozen q={q}")
    ok = np.all(v[ii,:q] & v[jj,:q],axis=1)
    x = (a[ii,:q,:6] - a[jj,:q,:6])/scale[None,None,:]
    motion = np.sqrt(np.mean(x*x,axis=(1,2)))
    grip_conflict = np.any((a[ii,:q,6]>0)!=(a[jj,:q,6]>0),axis=1)
    return ok,motion,grip_conflict


def cmd_screen(args):
    with open(args.frozen,encoding="utf-8") as f:
        fr=json.load(f)
    if fr.get("schema") != SCHEMA or fr.get("protocol_status") != "FROZEN_BASE_ONLY_NO_INCREMENTAL_RETUNING":
        raise ValueError("Unrecognized/unfrozen calibration protocol")
    if args.encoder_sha256.lower() != fr["calibration"]["encoder_checkpoint_sha256"]:
        raise ValueError("Frozen pre-FiLM image encoder SHA differs from calibration; cannot apply tau_v")
    d=load_records(args.records)
    pairs=get_pairs(args.pairs,d,require_labels=False)
    if any(int(d["task_ids"][i]) < args.first_incremental_task or
           int(d["task_ids"][j]) >= int(d["task_ids"][i]) for i,j,_ in pairs):
        raise ValueError("Screen requires ordered new task -> strictly earlier task pairs")
    p=fr["physical"]; v=fr["visual"]; ac=fr["action"]
    dd=pair_descriptors(d,pairs,p["pos_tolerance_m"],p["rot_tolerance_deg"],p["grip_tolerance_fraction"],
                        use_velocity=p["use_velocity"],
                        velocity_tol=p["velocity_tolerance_mps"],
                        angvelocity_tol=p["angular_velocity_tolerance_radps"])
    mvalid,motion,gconf=motion_pair(d,pairs,ac["q"],np.array(ac["motion_scale_base_IQR_floor_0p1"]))
    phys=dd["ds"] <= p["tau_s"]
    vis=dd["sv"] >= v["tau_v"]
    both=phys & vis & mvalid
    share=both & (motion <= ac["share_max_motion_distance"]) & ~gconf
    bif=both & ((motion >= ac["bif_min_motion_distance"]) | gconf)
    assert not np.any(share & bif)
    rows=[]
    for n,(i,j,_) in enumerate(pairs):
        status=("SHARE" if share[n] else "BIF" if bif[n] else
                "UNUSABLE_FUTURE_ACTION" if not mvalid[n] else
                "PHYSICAL_MISMATCH" if not phys[n] else
                "VISUAL_MISMATCH" if not vis[n] else "UNCERTAIN_ACTION")
        rows.append(dict(new_id=d["ids"][i],old_id=d["ids"][j],
                         new_task=int(d["task_ids"][i]),old_task=int(d["task_ids"][j]),
                         new_traj=str(d["traj_ids"][i]),old_traj=str(d["traj_ids"][j]),
                         state_distance=float(dd["ds"][n]),position_delta_m=float(dd["pos_m"][n]),
                         rotation_delta_deg=float(dd["rot_deg"][n]),
                         gripper_open_delta=float(dd["grip_delta"][n]),
                         velocity_delta_mps=(float(dd["velocity_delta_mps"][n]) if dd["velocity_delta_mps"] is not None else ""),
                         angular_velocity_delta_radps=(float(dd["angular_velocity_delta_radps"][n]) if dd["angular_velocity_delta_radps"] is not None else ""),
                         agent_cos=float(dd["agent_cos"][n]),eih_cos=float(dd["eih_cos"][n]),
                         min_visual_cos=float(dd["sv"][n]),motion_distance=float(motion[n]),
                         command_gripper_conflict=int(gconf[n]),future_valid=int(mvalid[n]),
                         label=status))
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; do not overwrite a previous audit")
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/"pair_audit.csv",rows,list(rows[0]))
    states={x:sum(r["label"]==x for r in rows) for x in
            ("SHARE","BIF","UNCERTAIN_ACTION","PHYSICAL_MISMATCH","VISUAL_MISMATCH","UNUSABLE_FUTURE_ACTION")}
    new_ids={r['new_id'] for r in rows}
    cov_share=len({r['new_id'] for r in rows if r['label']=='SHARE'}) / max(1,len(new_ids))
    cov_bif=len({r['new_id'] for r in rows if r['label']=='BIF'}) / max(1,len(new_ids))
    unique_trajectory_pairs=len(set((r['new_traj'],r['old_traj']) for r in rows))
    report={"status":"PHYSICAL_VISUAL_CANDIDATE_SCREEN_NOT_GROUND_TRUTH_SUCCESS",
            "n_unique_candidate_pairs":len(rows),"class_counts":states,
            "candidate_list_anchor_coverage_share":cov_share,
            "candidate_list_anchor_coverage_bif":cov_bif,
            "n_candidate_list_anchors":len(new_ids),
            "n_unique_trajectory_pairs":unique_trajectory_pairs,
            "limits":"Counts describe PROVIDED candidate list only. If candidates came from experiment A "
                     "top-k, coverage/recall over all actual windows cannot be inferred. "
                     "Human audit and independently stratified low-A-score negatives are required.",
            "input_sha256": {"records":sha256(args.records),"pairs":sha256(args.pairs),
                              "frozen_protocol":sha256(args.frozen)},
            "frozen_thresholds":{"tau_s":p['tau_s'],"tau_v":v['tau_v']}}
    dump_json(out/"audit_summary.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def cmd_inspect_h5(args):
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError("Install h5py only for inspect-h5: pip install h5py") from e
    with h5py.File(args.h5,"r") as h:
        print("HDF5",str(args.h5),"sha256",sha256(args.h5))
        seen=[0]
        def walk(name,obj):
            if seen[0]>=args.limit:
                return
            seen[0]+=1
            if isinstance(obj,h5py.Dataset):
                print("DATASET",name,"shape=",obj.shape,"dtype=",obj.dtype)
            else:
                print("GROUP",name)
        h.visititems(walk)
        if seen[0]>=args.limit:
            print(f"Truncated at --limit {args.limit}; increase only if necessary")
    print("Inspect actual keys. Do not assume robot0_eef_pos, robot0_eef_quat, or gripper keys exist.")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=parser.add_subparsers(dest="command",required=True)
    p=subs.add_parser("inspect-h5",help="List HDF5 datasets and shapes (no guessed field names)")
    p.add_argument("--h5",required=True)
    p.add_argument("--limit",type=int,default=120)
    p.set_defaults(func=cmd_inspect_h5)

    p=subs.add_parser("calibrate",help="Freeze state tolerances and visual threshold using BASE task human labels")
    p.add_argument("--records",required=True)
    p.add_argument("--pairs",required=True,help="CSV id_a,id_b,label (1=match; 0=nonmatch)")
    p.add_argument("--out",required=True)
    p.add_argument("--encoder-sha256",required=True,help="SHA256 of frozen pre-FiLM image encoder checkpoint")
    p.add_argument("--max-base-task",type=int,default=5)
    p.add_argument("--pos-m",type=float,default=DEFAULTS["position_m"])
    p.add_argument("--rot-deg",type=float,default=DEFAULTS["orientation_deg"])
    p.add_argument("--grip-frac",type=float,default=DEFAULTS["gripper_open_fraction"])
    p.add_argument("--velocity-mps",type=float,default=DEFAULTS["linear_velocity_mps"])
    p.add_argument("--angular-velocity-radps",type=float,default=DEFAULTS["angular_velocity_radps"])
    p.add_argument("--negative-quantile",type=float,default=DEFAULTS["fpr_quantile"])
    p.add_argument("--q",type=int,default=3)
    p.add_argument("--motion-share",type=float,default=DEFAULTS["motion_share"])
    p.add_argument("--motion-bif",type=float,default=DEFAULTS["motion_bif"])
    p.add_argument("--min-pos",type=int,default=20)
    p.add_argument("--min-neg",type=int,default=40)
    p.set_defaults(func=cmd_calibrate)

    p=subs.add_parser("screen",help="Apply FROZEN matching rule to incremental new->old pairs")
    p.add_argument("--records",required=True)
    p.add_argument("--pairs",required=True,help="CSV id_a,id_b OR new_file,old_file")
    p.add_argument("--frozen",required=True)
    p.add_argument("--encoder-sha256",required=True,help="Must match encoder hash in frozen.json")
    p.add_argument("--out-dir",required=True)
    p.add_argument("--first-incremental-task",type=int,default=6)
    p.set_defaults(func=cmd_screen)
    args=parser.parse_args()
    if args.command=="calibrate":
        if not (0<args.negative_quantile<1) or args.q<1 or args.motion_share>=args.motion_bif:
            parser.error("Invalid calibration quantile / action horizon / action distance thresholds")
    args.func(args)


if __name__=="__main__":
    main()
