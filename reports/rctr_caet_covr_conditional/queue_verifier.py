"""External read-only acceptance. Never imports a policy, simulator or optimizer."""
import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def exact(v):
    return "%d/%d" % (v.numerator, v.denominator)


def live_workers(output):
    """Catch spawned environment children even when argv lacks the method name."""
    matches = []
    expected = b"CORESET_OUT="+str(output).encode()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int(proc.name) in (os.getpid(), os.getppid()):
                continue
            if expected in (proc/"environ").read_bytes().split(b"\0"):
                stat = (proc/"stat").read_text().split(") ", 1)[1].split()
                if stat[0] != "Z":
                    matches.append(int(proc.name))
        except (OSError, ValueError):
            continue
    return matches


def matrix(root, seed):
    directory = root/"ra_coreset/evaluation"
    files = sorted(directory.glob("trials_stage*_task*.jsonl"))
    expected = {"trials_stage%d_task%d.jsonl" % (s,t)
                for s in range(6,10) for t in range(6,s+1)}
    assert {p.name for p in files} == expected
    rates, records, digest = {}, {}, hashlib.sha256()
    for f in files:
        digest.update(f.name.encode()+b"\0"+bytes.fromhex(sha(f)))
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        stage, task = int(f.stem.split("_")[1][5:]), int(f.stem.split("_")[2][4:])
        assert len(rows) == 20 and {r["trial"] for r in rows} == set(range(20))
        assert all(r["stage"] == stage and r["task"] == task and
                   r["seed"] == seed and r["arm"] == "ra_coreset" and
                   type(r["success"]) is bool for r in rows)
        rates[stage,task] = Fraction(sum(r["success"] for r in rows), 20)
        for r in rows:
            records[stage,task,r["trial"]] = r
    auc = sum((sum(rates[s,t] for s in range(t,10))/ (10-t)
               for t in range(6,10)), Fraction(0))/4
    return auc, records, digest.hexdigest()


def validate_artifacts(root, source, seed, steps, baseline=False):
    audit = read(root/"AUDIT.json")
    assert audit["status"] == "PASS"
    assert audit["epochs"] == 400 and audit["optimizer_steps"] == steps
    assert audit["formal_records"] == 200 and audit["vector_attempts"] == 20
    checkpoint_rows = audit["checkpoints"]
    assert len(checkpoint_rows) == 4
    checkpoint_sha = {}
    for row in checkpoint_rows:
        task = row["task"]
        assert task in range(6,10)
        assert row.get("strict_loaded_keys", row.get("loaded_keys", 0)) > 0
        p = root/"ra_coreset/models"/("mlrifa_%d_model_ep99.pth" % task)
        assert sha(p) == row["sha256"]
        checkpoint_sha[task] = row["sha256"]
    assert len(checkpoint_sha) == 4
    auc, records, digest = matrix(root, seed)
    assert all(r["checkpoint_sha256"] == checkpoint_sha[r["stage"]]
               for r in records.values())
    events = Counter()
    for f in (root/"ra_coreset/evaluation").glob("attempts*.jsonl"):
        events.update(json.loads(line)["event"] for line in f.read_text().splitlines())
    assert events == Counter(vector_attempt_start=20,vector_attempt_complete=20,
        environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
    epochs = read(root/"ra_coreset/epochs.json")
    assert len(epochs) == 400
    assert {(r["task"],r["epoch"]) for r in epochs} == {
        (t,e) for t in range(6,10) for e in range(100)}
    assert sum(r["batches"] for r in epochs) == epochs[-1]["steps"] == steps
    if not baseline:
        contract = read(root/"CONTRACT.json")
        for name, digest_expected in contract["source_sha256"].items():
            assert sha(source/name) == digest_expected, name
        actual = {str(p.relative_to(source)) for p in source.rglob("*")
                  if p.is_file() and p.suffix in (".py",".yaml",".sh")
                  and "__pycache__" not in p.parts}
        assert actual == set(contract["source_sha256"])
        for name, digest_expected in contract["files"].items():
            assert sha(name) == digest_expected, name
        for task in range(6,10):
            manifest = read(root/"ra_coreset/storage"/("task%d.json" % task))
            for item in manifest["selected"]:
                assert sha(item["source"]) == item["sha256"]
                assert sha(root/"ra_coreset/buffer"/item["destination"]) == item["sha256"]
    return auc, records, digest


def verify_worker(entry):
    root, source, parent = map(Path, (entry["output"], entry["source"], entry["baseline"]))
    assert not live_workers(root), "Residual worker or environment process"
    for name in ("pipeline.exit", "supervisor.exit"):
        assert (root/name).read_text().strip() == "0", name
    acceptance = root/"PROCESS_EXIT_ACCEPTANCE.json"
    result = read(acceptance)
    assert result["status"] == "PASS" and result["pipeline_exit"] == 0
    assert result["audit_sha256"] == sha(root/"AUDIT.json")
    baseline_acceptance = Path(entry["baseline_acceptance"])
    assert read(baseline_acceptance)["status"] == "PASS"
    steps = 105500 if entry["suite"] == "OBJECT" else 76100
    auc, new, digest = validate_artifacts(root, source, entry["seed"], steps)
    base, old, base_digest = validate_artifacts(parent, None, entry["seed"], steps, True)
    for key in new:
        assert new[key]["initial_state_sha256"] == old[key]["initial_state_sha256"]
        assert new[key]["modal_sha256"] == old[key]["modal_sha256"]
    return dict(engineering="PASS", auc=exact(auc), baseline_auc=exact(base),
        acceptance_sha256=sha(acceptance), baseline_acceptance_sha256=sha(baseline_acceptance),
        episodes_sha256=digest, baseline_episodes_sha256=base_digest,
        read_only_independent_verification=True, formal_records=200,
        baseline_records=200, residual_workers=[])
