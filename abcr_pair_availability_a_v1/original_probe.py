#!/usr/bin/env python3
"""Pre-training feasibility probe for ABCR on existing MLR .pt files.

Stage A only: pairs identified with language-conditioned latent visual features
are CANDIDATES, not proof that physical states match.  Re-export proprioception,
unmodulated image embeddings, and post-anchor action snippets for Stage B.

Examples:
python abcr_pair_probe.py --feature-dir /data/features --buffer-dir /data/buffer --current-task 7 --out-dir ./probe_T7
python abcr_pair_probe.py --feature-dir /data/features --buffer-dir /data/buffer --current-task 7 --past-q 4 --out-dir ./probe_T7_past4
"""
import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def _feature_and_actions(path, past_q):
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "feature" not in payload or "actions" not in payload:
        raise ValueError(f"{path}: expected dict with 'feature' and 'actions'")
    h = torch.as_tensor(payload["feature"], dtype=torch.float32)
    a = torch.as_tensor(payload["actions"], dtype=torch.float32)
    while h.ndim > 3 and h.shape[0] == 1:
        h = h.squeeze(0)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a.squeeze(0)
    if h.ndim != 3:
        raise ValueError(f"{path}: feature must be [T,M,E], got {tuple(h.shape)}")
    if h.shape[1] == 4:  # repository default: [T,4,E]
        pass
    elif h.shape[0] == 4:  # alternative layout: [4,T,E]
        h = h.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"{path}: cannot identify feature modalities in {tuple(h.shape)}")
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if a.ndim != 2 or a.shape[-1] != 7:
        raise ValueError(f"{path}: expected continuous action vectors with 7 dimensions; got {tuple(a.shape)}")
    if len(a) < past_q:
        raise ValueError(f"{path}: action window length {len(a)} < --past-q {past_q}")
    if len(h) < 1:
        raise ValueError(f"{path}: empty latent sequence")
    # The published repository uses indices 0=language, 1=agent view,
    # 2=eye-in-hand.  Both visual slots remain FiLM/language-conditioned.
    v1 = F.normalize(h[-1, 1, :], dim=0, eps=1e-8)
    v2 = F.normalize(h[-1, 2, :], dim=0, eps=1e-8)
    desc = torch.cat((v1, v2), dim=0) / math.sqrt(2.0)
    return desc, a[-past_q:].clone()


def _load(paths, q, warn_limit=5):
    descs, actions, kept, dropped = [], [], [], []
    for path in paths:
        try:
            v, a = _feature_and_actions(path, q)
            descs.append(v)
            actions.append(a)
            kept.append(str(path))
        except Exception as ex:
            dropped.append((str(path), str(ex)))
    if dropped:
        print(f"[warning] skipped {len(dropped)} unusable files:")
        for p, e in dropped[:warn_limit]:
            print("   ", p, e)
    if not kept:
        raise RuntimeError("No usable .pt samples")
    return torch.stack(descs), torch.stack(actions), kept, dropped


def _action_stats(a_new, a_old):
    # Each action value usually lives in roughly [-1,1]; use robust scales.
    all_mot = torch.cat([a_new[:, :, :6], a_old[:, :, :6]], 0).flatten(0, 1)
    qs = torch.quantile(all_mot, torch.tensor([0.25, 0.75]), dim=0)
    return (qs[1] - qs[0]).clamp_min(0.1)


def _similarities(nv, ov, na, oa, scale):
    """Pairwise visual similarity, normalized motion discrepancy, grip conflict, idle mask."""
    visual = nv @ ov.T
    dmot = torch.sqrt(((((na[:, None, :, :6] - oa[None, :, :, :6]) / scale) ** 2).mean(dim=(2, 3))).clamp_min(0.0))
    grip_conflict = ((na[:, None, :, 6] > 0) != (oa[None, :, :, 6] > 0)).any(dim=-1)
    speed_n = na[:, :, :6].square().mean(dim=(1, 2)).sqrt()
    speed_o = oa[:, :, :6].square().mean(dim=(1, 2)).sqrt()
    idle = (speed_n[:, None] < 0.05) & (speed_o[None, :] < 0.05)
    return visual, dmot, grip_conflict, idle


def _f(x):
    return round(float(x), 6)


def _safe_div(n, d):
    return _f(n / d) if d else None


def _write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args):
    rng = random.Random(args.seed)
    current_dir = Path(args.feature_dir) / str(args.current_task)
    old_dir = Path(args.buffer_dir)
    new_paths = sorted(current_dir.glob("*.pt"))
    if not new_paths:
        raise FileNotFoundError(f"No current task features under {current_dir}")
    old_groups = defaultdict(list)
    for p in sorted(old_dir.glob("*.pt")):
        m = re.match(r"t(\d+)__", p.name)
        if not m:
            continue
        old_id = int(m.group(1))
        if old_id == args.current_task:
            continue
        if old_id > args.current_task:
            raise ValueError(f"Unexpected future-task file in old replay buffer: {p.name}")
        old_groups[old_id].append(p)
    if not old_groups:
        raise FileNotFoundError("No task-prefixed old .pt files found in buffer directory")
    exact_files = {"new_total": len(new_paths), "old_per_task_total": {str(k): len(v) for k, v in old_groups.items()}}
    if args.max_current and len(new_paths) > args.max_current:
        new_paths = rng.sample(new_paths, args.max_current)
    old_paths = []
    for task, paths in sorted(old_groups.items()):
        if args.max_old_per_task and len(paths) > args.max_old_per_task:
            paths = rng.sample(paths, args.max_old_per_task)
        old_paths += [(task, p) for p in paths]
    old_task_id_all = [task for task, path in old_paths]
    new_v, new_a, new_kept, new_drop = _load([p for p in new_paths], args.past_q)
    old_v, old_a, old_kept, old_drop = _load([p for task, p in old_paths], args.past_q)
    old_paths_to_task = {str(path): tid for tid, path in old_paths}
    old_tasks = torch.tensor([old_paths_to_task[path] for path in old_kept], dtype=torch.long)
    scale = _action_stats(new_a, old_a)
    thr_values = sorted(set(float(x) for x in args.visual_thresholds.split(",")))
    for v in thr_values:
        if not -1 <= v <= 1:
            raise ValueError("cosine similarity thresholds must be in [-1,1]")
    if args.action_share >= args.action_bif:
        raise ValueError("action-share threshold must be < action-bif threshold")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    out_rows = []
    per_old_rows = []
    audit_rows = []
    n_new, n_old = len(new_kept), len(old_kept)
    total_possible = n_new * n_old
    for threshold in thr_values:
        counts = defaultdict(int)
        per_task = {tid: defaultdict(int) for tid in old_groups}
        has_share = torch.zeros(n_new, dtype=torch.bool)
        has_bif = torch.zeros(n_new, dtype=torch.bool)
        top_share = torch.zeros(n_new, dtype=torch.bool)
        top_bif = torch.zeros(n_new, dtype=torch.bool)
        top_pos_ct = 0
        top_neg_ct = 0
        top_pairs = 0
        for start in range(0, n_new, args.chunk):
            stop = min(start + args.chunk, n_new)
            sv, d, gc, idle = _similarities(new_v[start:stop], old_v,
                                            new_a[start:stop], old_a, scale)
            cand = sv >= threshold
            sp = cand & (d <= args.action_share) & ~gc
            bn = cand & ((d >= args.action_bif) | gc)
            assert not (sp & bn).any()
            counts["candidate"] += int(cand.sum())
            counts["share"] += int(sp.sum())
            counts["bif"] += int(bn.sum())
            counts["share_idle"] += int((sp & idle).sum())
            has_share[start:stop] = sp.any(dim=1)
            has_bif[start:stop] = bn.any(dim=1)
            for tid in sorted(old_groups):
                col = (old_tasks == tid)
                per_task[tid]["raw"] += (stop - start) * int(col.sum())
                per_task[tid]["candidate"] += int(cand[:, col].sum())
                per_task[tid]["share"] += int(sp[:, col].sum())
                per_task[tid]["bif"] += int(bn[:, col].sum())
            k = min(args.topk, n_old)
            idx = torch.topk(sv, k=k, dim=1).indices
            tr = torch.gather(cand, 1, idx)
            ts = torch.gather(sp, 1, idx)
            tb = torch.gather(bn, 1, idx)
            ti = torch.gather(idle, 1, idx)
            td = torch.gather(d, 1, idx)
            tv = torch.gather(sv, 1, idx)
            top_pairs += int(tr.sum())
            top_pos_ct += int(ts.sum())
            top_neg_ct += int(tb.sum())
            top_share[start:stop] = ts.any(dim=1)
            top_bif[start:stop] = tb.any(dim=1)
            # Keep all *retrieved* candidates for transparent human auditing.
            for i in range(stop - start):
                for z in range(k):
                    if not bool(tr[i, z]):
                        continue
                    oj = int(idx[i, z])
                    cls = "share" if bool(ts[i, z]) else "bif" if bool(tb[i, z]) else "uncertain"
                    audit_rows.append({
                        "visual_threshold": threshold,
                        "class": cls,
                        "new_file": new_kept[start + i],
                        "old_file": old_kept[oj],
                        "old_task": int(old_tasks[oj]),
                        "visual_cosine": _f(tv[i, z]),
                        "action_distance": _f(td[i, z]),
                        "both_idle": int(ti[i, z])
                    })
        row = {
            "visual_threshold": threshold,
            "n_current_samples": n_new,
            "n_old_samples": n_old,
            "all_possible_cross_pairs": total_possible,
            "visual_candidates": counts["candidate"],
            "share_pairs": counts["share"],
            "bif_pairs": counts["bif"],
            "share_idle_pairs": counts["share_idle"],
            "candidate_rate_all": _safe_div(counts["candidate"], total_possible),
            "share_rate_all": _safe_div(counts["share"], total_possible),
            "bif_rate_all": _safe_div(counts["bif"], total_possible),
            "share_given_match": _safe_div(counts["share"], counts["candidate"]),
            "bif_given_match": _safe_div(counts["bif"], counts["candidate"]),
            "share_nonidle_fraction": _safe_div(counts["share"] - counts["share_idle"], counts["share"]),
            "anchor_share_coverage_all": _f(has_share.float().mean()),
            "anchor_bif_coverage_all": _f(has_bif.float().mean()),
            "retrieved_topk_matched": top_pairs,
            "retrieved_topk_share": top_pos_ct,
            "retrieved_topk_bif": top_neg_ct,
            "anchor_share_coverage_retrieved": _f(top_share.float().mean()),
            "anchor_bif_coverage_retrieved": _f(top_bif.float().mean()),
        }
        # Balanced 5+5 microbatch feasibility, without mining/retrieval.
        b_new = min(args.batch_new, n_new)
        b_old = min(args.batch_old, n_old)
        step_s, step_b, step_both = [], [], []
        for _ in range(args.batch_trials):
            ci = torch.tensor(rng.sample(range(n_new), b_new))
            oi = torch.tensor(rng.sample(range(n_old), b_old))
            vs, dist, gc, _ = _similarities(new_v[ci], old_v[oi],new_a[ci],old_a[oi], scale)
            match = vs >= threshold
            np = int((match & (dist <= args.action_share) & ~gc).sum())
            nn = int((match & ((dist >= args.action_bif) | gc)).sum())
            step_s.append(np)
            step_b.append(nn)
            step_both.append(np >= 1 and nn >= 1)
        row.update({
            "mean_share_per_balanced_batch": _f(sum(step_s)/len(step_s)),
            "mean_bif_per_balanced_batch": _f(sum(step_b)/len(step_b)),
            "fraction_batches_with_share": _f(sum(v >= 1 for v in step_s)/len(step_s)),
            "fraction_batches_with_bif": _f(sum(v >= 1 for v in step_b)/len(step_b)),
            "fraction_batches_with_both": _f(sum(step_both)/len(step_both)),
        })
        out_rows.append(row)
        for tid, c in sorted(per_task.items()):
            per_old_rows.append({
                "visual_threshold":threshold, "old_task":tid,
                "possible":c["raw"], "matched":c["candidate"],
                "share":c["share"],"bif":c["bif"],
                "matched_rate": _safe_div(c["candidate"],c["raw"]),
                "share_cond_match": _safe_div(c["share"], c["candidate"]),
                "bif_cond_match": _safe_div(c["bif"], c["candidate"]),
            })
    _write_csv(root / "summary.csv", out_rows, list(out_rows[0]))
    _write_csv(root / "by_old_task.csv", per_old_rows, list(per_old_rows[0]))
    if audit_rows:
        _write_csv(root / "retrieved_candidates.csv", audit_rows, list(audit_rows[0]))
        # Group-aware stratified sample for manual inspection, not statistically representative.
        samples = []
        buckets = defaultdict(list)
        for x in audit_rows:
            buckets[(x["visual_threshold"], x["class"],x["old_task"])].append(x)
        for k, v in sorted(buckets.items(), key=lambda e:str(e[0])):
            samples.extend(rng.sample(v, min(args.audit_per_bucket, len(v))))
        _write_csv(root / "audit_sample.csv", samples, list(samples[0]))
    report = {
        "status":"screening_only_not_ground_truth_state_matching",
        "data_limitations": ["visual features may be FiLM conditioned on task language",
                             "no unmodulated state, trajectory_id, or forward-looking actions in original pt",
                             "past-q>1 compares past action windows, not future actions",
                             "sample windows are not independent demonstrations"],
        "n_current_used":n_new,"n_old_used":n_old,
        "raw_file_counts":exact_files,
        "dropped_current":len(new_drop),"dropped_old":len(old_drop),
        "used_old_task_counts":{str(t): int((old_tasks==t).sum()) for t in sorted(old_groups)},
        "scale_motion_iqr_floor_0p1": [float(x) for x in scale],
        "parameters":vars(args),
    }
    (root/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nGenerated:", root)
    print("Raw files (current):", len(new_paths), "used:",n_new,
          "| raw files (old):",len(old_paths),"used:",n_old)
    print("Important: metrics are SCREENING PROXIES, not true state-comparable ground truth.")
    for row in out_rows:
        print("sim>=",row["visual_threshold"],"candidate",row["visual_candidates"],
              "share",row["share_pairs"],"bif",row["bif_pairs"],
              "share_anchor_topk",row["anchor_share_coverage_retrieved"],
              "bif_anchor_topk",row["anchor_bif_coverage_retrieved"],
              "both_per_batch",row["fraction_batches_with_both"])


def main():
    p=argparse.ArgumentParser(description="ABCR original-replay pair availability screen; NO TRAINING")
    p.add_argument("--feature-dir",required=True,help="root containing CURRENT_TASK/*.pt")
    p.add_argument("--buffer-dir",required=True,help="flat old replay directory with tTASK__*.pt names")
    p.add_argument("--current-task",type=int,required=True)
    p.add_argument("--out-dir",default="./abcr_pair_probe_output")
    p.add_argument("--past-q",type=int,default=1,help="1=current action; >=2=PAST action snippet only")
    p.add_argument("--visual-thresholds",default="0.7,0.8,0.9,0.95",help="threshold sweep, NOT calibrated physics")
    p.add_argument("--action-share",type=float,default=0.5,help="dimension-normalized motion distance; heuristic")
    p.add_argument("--action-bif",type=float,default=1.5,help="dimension-normalized motion distance; heuristic")
    p.add_argument("--topk",type=int,default=5)
    p.add_argument("--batch-new",type=int,default=5)
    p.add_argument("--batch-old",type=int,default=5)
    p.add_argument("--batch-trials",type=int,default=300)
    p.add_argument("--max-current",type=int,default=0,help="0=all; positive=sample capped count")
    p.add_argument("--max-old-per-task",type=int,default=0,help="0=all; positive=sample capped count")
    p.add_argument("--audit-per-bucket",type=int,default=15)
    p.add_argument("--chunk",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.past_q <= 0 or args.topk<=0 or args.batch_new<=0 or args.batch_old<=0 or args.batch_trials<=0 or args.chunk<=0:
        p.error("past-q/topk/batch-new/batch-old/batch-trials/chunk must be positive")
    run(args)

if __name__ == "__main__":
    main()
