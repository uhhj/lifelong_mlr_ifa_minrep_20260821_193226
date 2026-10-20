"""Bounded current-stage RCTR collection. No optimizer; no historical-state replay."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from coreset_exp.common import O, SEED, verify, sha, write, save
from coreset_exp.model import cfg, data_setup, complete, strict_downstream
from libero.lifelong.metric import OffScreenRenderEnv
from libero.lifelong.utils import control_seed
from . import snapshot_runtime as rt


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


@torch.no_grad()
def native_feature(policy, history, instr, mask, c):
    batch = rt.batch_of(history, instr, mask, c)
    prepared = policy.preprocess_input(batch, train_mode=False)
    x = policy.spatial_encode(prepared)
    assert tuple(x.shape) == (1, 8, 4, 512)
    return x[0].detach().cpu().clone()


def run_branch(env, policy, snap, initial, at, phase, rep, cand,
               instr, mask, c, cache, private):
    key = "%d_%d_%s_%d_%s" % (initial, at, phase, rep, cand["id"])
    if key in cache:
        row = cache[key]
        saved = private/(key+".pth")
        payload = torch.load(saved, map_location="cpu") if row.get("payload_sha256") else None
        if payload is not None:
            assert sha(saved) == row["payload_sha256"]
        return row, payload
    obs, history, _ = rt.restore(env, policy, snap)
    noise_seed = 100000000*SEED + 1000000*c.current_task + 1000*initial + 2*at + rep
    if phase == "confirmation":
        noise_seed += 500000
    table = rt.noise(noise_seed, 600-at)
    identity = dict(key=key, task=c.current_task, initial=initial, snapshot_step=at,
                    phase=phase, repeat=rep, candidate=cand,
                    noise_seed=noise_seed, noise_sha256=rt.digest(table))
    rt.ledger("branch_start", **identity)
    begun = time.monotonic()
    payload = None
    success = False
    skipped = None
    steps = 0
    for i in range(600-at):
        history = rt.stack(history, obs)
        par, winner = rt.params(policy, rt.batch_of(history, instr, mask, c))
        action = rt.normal(par, winner, table[i])
        if i == 0 and cand["axis"] is not None:
            baseline = np.clip(action, -1, 1)
            action = baseline.copy()
            action[cand["axis"]] = np.clip(action[cand["axis"]]+.05*cand["sign"], -1, 1)
            if np.linalg.norm(action-baseline) <= 1e-7:
                skipped = "ZERO_INTERVENTION"
                break
        if i == 1:
            component = winner
            if cand["role"] != "original":
                ranking = sorted((j for j in range(5) if j != winner),
                                 key=lambda j: (-float(par[2][j]), j))
                component = ranking[int(cand["role"])-1]
                action = par[0][component].copy()
            payload = dict(feature=native_feature(policy, history, instr, mask, c),
                           action=torch.from_numpy(action.copy()),
                           physical_sha256=rt.digest(env.get_sim_state()),
                           component=component, winner=winner)
        obs, success = rt.envstep(env, action, "screening")
        steps += 1
        if success:
            break
    item = dict(**identity, success=bool(success) if skipped is None else None,
                status="SKIPPED" if skipped else "COMMITTED", skip_reason=skipped,
                steps=steps, seconds=time.monotonic()-begun)
    if payload is not None:
        save(private/(key+".pth"), payload)
        item["payload_sha256"] = sha(private/(key+".pth"))
    # Atomic per-key commit is the authority. The append-only ledgers are mirrors;
    # technical recovery must never repeat a committed failure or success.
    write(rt.O/"committed"/(key+".json"), item)
    rt.ledger("branch_complete", **item)
    with (rt.O/"episodes.jsonl").open("a") as f:
        f.write(json.dumps(item)+"\n")
        f.flush()
        os.fsync(f.fileno())
    cache[key] = item
    rt.COUNTS["completed_branches"] += 1
    rt.status("COLLECTING", task=c.current_task, key=key)
    return item, payload


def candidate(axis=None, sign=0, role="original"):
    return dict(id="normal" if axis is None else "a%d_%+d_%s" % (axis, sign, role),
                axis=axis, sign=sign, role=role)


def qualifies(parent, child):
    if any(r[0]["success"] is None for r in parent+child):
        return False
    fs = sum(not a[0]["success"] and b[0]["success"] for a, b in zip(parent, child))
    sf = sum(a[0]["success"] and not b[0]["success"] for a, b in zip(parent, child))
    return fs >= 1 and sf == 0


def process_snapshot(env, policy, snap, initial, at, instr, mask, c, cache, private):
    def run(phase, rep, cand):
        return run_branch(env, policy, snap, initial, at, phase, rep, cand,
                          instr, mask, c, cache, private)
    normal = [run("exploration", r, candidate()) for r in range(2)]
    alternatives = []
    for axis in range(3):
        for sign in (-1, 1):
            base = [run("exploration", r, candidate(axis, sign)) for r in range(2)]
            for role in ("1", "2"):
                cand = candidate(axis, sign, role)
                result = [run("exploration", r, cand) for r in range(2)]
                if qualifies(base, result):
                    alternatives.append((cand, base, result))
    # Fixed axis/sign/nonwinner-rank order. Only the first qualified candidate gets
    # confirmation; a failed confirmation never causes another candidate to be tried.
    if not alternatives:
        return None
    cand, base, result = alternatives[0]
    confirm_base = [run("confirmation", r, candidate(cand["axis"], cand["sign"]))
                    for r in range(2)]
    confirm_candidate = [run("confirmation", r, cand) for r in range(2)]
    if not qualifies(confirm_base, confirm_candidate):
        return None
    for rep in range(2):
        if not base[rep][0]["success"] and result[rep][0]["success"]:
            plus = result[rep][1]
            minus = base[rep][1]
            assert plus is not None and minus is not None
            assert plus["physical_sha256"] == minus["physical_sha256"]
            assert torch.equal(plus["feature"], minus["feature"])
            has_normal = bool(normal[rep][0]["success"]) and normal[rep][1] is not None
            return dict(certified=True, task=c.current_task, initial=initial,
                snapshot_step=at, candidate=cand, exploration_repeat=rep,
                zdelta=plus["feature"], a_plus=plus["action"], a_minus=minus["action"],
                has_normal_reference=has_normal,
                z0=normal[rep][1]["feature"] if has_normal else None,
                a0=normal[rep][1]["action"] if has_normal else None,
                evidence_keys=[x[0]["key"] for x in base+result+confirm_base+confirm_candidate+normal])
    raise AssertionError("Certified pair must contain an exploration rescue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=(6, 7, 8))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    contract = verify()
    root = O/"rctr"/("task%d" % args.task)
    if (root/"ACCEPTANCE.json").exists():
        assert json.loads((root/"ACCEPTANCE.json").read_text())["status"] == "PASS"
        return
    assert args.resume or not root.exists(), "Explicit technical resume required"
    root.mkdir(parents=True, exist_ok=True)
    rt.initialize(root)
    private = root/"private"
    private.mkdir(exist_ok=True)
    cache = {p.stem: json.loads(p.read_text()) for p in (root/"committed").glob("*.json")}
    for r in rows(root/"episodes.jsonl"):
        assert r["key"] in cache and cache[r["key"]] == r
    cp = O/"ra_coreset/models"/("mlrifa_%d_model_ep99.pth" % args.task)
    cp_hash = sha(cp)
    torch.backends.cudnn.benchmark = True
    control_seed(SEED)
    c = cfg("eval")
    c.current_task = args.task
    bench, _, shape = data_setup(c, False)
    c.shape_meta = shape
    policy = complete(c, shape)
    strict_downstream(policy, cp)
    policy.eval().requires_grad_(False)
    task = bench.get_task(args.task)
    instr, mask = bench.get_task_emb(args.task)
    states = torch.load(Path(c.init_states_folder)/task.problem_folder/task.init_states_file)
    assert len(states) >= 26
    inventory_path = root/"SNAPSHOTS.json"
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    certificates = {}
    existing_cert = root/"partial_certificates.pth"
    if existing_cert.exists():
        certificates = torch.load(existing_cert, map_location="cpu")
    for index in range(20, 26):
        if all("%d_%d" % (index, at) in certificates for at in (40, 120)):
            continue
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
            camera_heights=c.data.img_h, camera_widths=c.data.img_w)
        try:
            env.seed(SEED)
            env.reset()
            env.seed(SEED)
            policy.reset()
            obs = env.set_init_state(states[index])
            history = []
            done = False
            for _ in range(5):
                obs, success = rt.envstep(env, np.zeros(7), "prefix")
                done = done or success
            table = rt.noise(SEED*10000000 + args.task*10000 + index, 120)
            snaps = {}
            for t in range(120):
                if done:
                    break
                history = rt.stack(history, obs)
                par, winner = rt.params(policy, rt.batch_of(history, instr, mask, c))
                obs, done = rt.envstep(env, rt.normal(par, winner, table[t]), "prefix")
                if t+1 in (40, 120) and not done:
                    snaps[t+1] = rt.snapshot(env, policy, obs, history[-7:])
            for at in (40, 120):
                key = "%d_%d" % (index, at)
                if key in certificates:
                    continue
                if at not in snaps:
                    certificates[key] = None
                    continue
                snap = snaps[at]
                identity = dict(physical=rt.digest(snap["physical"]),
                    observations={k: rt.digest(v) for k, v in snap["obs"].items()
                                  if isinstance(v, np.ndarray)},
                    initial=rt.digest(states[index]), prefix_noise=rt.digest(table))
                if key in inventory:
                    assert identity == inventory[key], "Technical recollection differs"
                else:
                    inventory[key] = identity
                    write(inventory_path, inventory)
                # Native action/history and complete state+observation repeat gates.
                gate = rt.gate(env, policy, snap, index, instr, mask, c)
                write(root/("gate_%s_%d.json" % (key, time.time_ns())), gate)
                certificates[key] = process_snapshot(env, policy, snap, index, at,
                                                      instr, mask, c, cache, private)
                save(existing_cert, certificates)
            save(existing_cert, certificates)
        finally:
            env.close()
    records = [certificates[k] for k in sorted(certificates)
               if certificates[k] is not None]
    assert len(certificates) == 12 and len(records) <= 12
    # Unique scientific branches are bounded; interrupted technical attempts are
    # separate ledger entries, never an excuse to re-run a committed normal failure.
    assert sum(r["phase"] == "exploration" for r in cache.values()) <= 456
    assert sum(r["phase"] == "confirmation" for r in cache.values()) <= 48
    assert all(r["steps"] <= 600-r["snapshot_step"] for r in cache.values())
    save(root/"sidecar.pth", records)
    from .validate_sidecar import validate
    gradient = validate(root/"sidecar.pth", cp)
    assert sha(cp) == cp_hash
    verify()
    write(root/"ACCEPTANCE.json", dict(status="PASS", task=args.task,
        records=len(records), sidecar_sha256=sha(root/"sidecar.pth"),
        sidecar_bytes=(root/"sidecar.pth").stat().st_size,
        unique_branches=len(cache), all_attempts=len(rows(root/"attempts.jsonl")),
        checkpoint_sha256=cp_hash, invocation_counts=rt.COUNTS,
        optimizer_calls=0, contract_sha256=sha(O/"CONTRACT.json"),
        real_record_gradient=gradient,
        certificate_metadata=[{k: v for k, v in r.items()
                              if k not in ("zdelta", "z0", "a_plus", "a_minus", "a0")}
                              for r in records]))


if __name__ == "__main__":
    main()
