"""Independent frozen run accounting. No environment calls and no optimizer."""
import json
import os
from pathlib import Path
from collections import Counter
import torch
from coreset_exp.common import O, verify, sha, hashes, write
from coreset_exp.model import cfg, strict_downstream
from libero.lifelong.models.brief import BriefPolicy
from ra_exp.audit import matrix, metrics, pack


def js(p):
    return json.loads(p.read_text())


def main():
    verify()
    parent = Path(os.environ["RA_BASE"])
    root = O/"ra_coreset"
    history = js(root/"epochs.json")
    assert len(history) == 400
    assert {(r["task"], r["epoch"]) for r in history} == {
        (t, e) for t in range(6, 10) for e in range(100)}
    assert sum(r["batches"] for r in history) == history[-1]["steps"] == 76100
    assert js(O/"TASK6_TENSOR_PARITY.json")["equal"]
    assert js(root/"INITIAL_STATE.json") == js(parent/"ra_coreset/INITIAL_STATE.json")
    assert js(O/"GPU_PREFLIGHT.json")["status"] == "PASS"
    policy = BriefPolicy(cfg("ra_coreset"), None).cpu()
    checkpoints, selections, ifa = [], {}, {}
    for task in range(6, 10):
        cp = root/"models"/("mlrifa_%d_model_ep99.pth" % task)
        raw = torch.load(cp, map_location="cpu")
        assert raw["task"] == task and raw["epoch_index"] == 99
        assert raw["contract_sha256"] == sha(O/"CONTRACT.json")
        checkpoints.append(dict(task=task, sha256=sha(cp),
            strict_loaded_keys=strict_downstream(policy, cp)))
        current = js(root/"storage"/("task%d.json" % task))
        previous = js(parent/"ra_coreset/storage"/("task%d.json" % task))
        assert current["K"] == previous["K"]
        assert current["selection"]["selected_ids"] == previous["selection"]["selected_ids"]
        assert [(Path(x["source"]).name, x["destination"], x["sha256"])
                for x in current["selected"]] == [
            (Path(x["source"]).name, x["destination"], x["sha256"]) for x in previous["selected"]]
        for x in current["selected"]:
            assert sha(x["source"]) == sha(root/"buffer"/x["destination"]) == x["sha256"]
        selections[str(task)] = dict(ordered_ids_equal=True,
            payloads_equal=True, count=current["K"])
        ifa[str(task)] = dict(equal=js(root/("ifa_task%d.json" % task)) ==
                              js(parent/"ra_coreset"/("ifa_task%d.json" % task)))
        assert ifa[str(task)]["equal"], "Same RA inputs must give the same IFA pairs"
    assert hashes(root/"buffer") == js(root/"buffer_after_task9.json")["hashes"]
    from .audit_effects import audit as effect_audit
    auxiliary = effect_audit()
    seen_steps=[]
    for row in history:
        stats=js(root/"auxiliary"/("task%d_epoch%02d.json"%(row["task"],row["epoch"])))
        assert len(stats)==row["batches"]
        seen_steps.extend(x["step"] for x in stats)
        if row["task"]==6:
            assert all(x["effect_loss"] is None for x in stats)
    assert seen_steps==list(range(76100))
    rates, new = matrix(O, "ra_coreset")
    base, old = matrix(parent, "ra_coreset")
    for key, row in new.items():
        assert row["initial_state_sha256"] == old[key]["initial_state_sha256"]
        assert row["modal_sha256"] == old[key]["modal_sha256"]
        assert row["checkpoint_sha256"] == checkpoints[row["stage"]-6]["sha256"]
    counts = Counter()
    for p in (root/"evaluation").glob("attempts*.jsonl"):
        counts.update(json.loads(line)["event"] for line in p.read_text().splitlines())
    expected = Counter(vector_attempt_start=20, vector_attempt_complete=20,
        environment_construct_start=10, environment_construct_complete=10, environment_closed=10)
    assert counts == expected
    current_metrics, parent_metrics = metrics(rates), metrics(base)
    verify()
    write(O/"AUDIT.json", dict(status="PASS", method="CAET", epochs=400,
        optimizer_steps=76100, formal_records=200, vector_attempts=20,
        checkpoints=checkpoints, selection=selections, ifa=ifa, auxiliary=auxiliary,
        all_inputs_unchanged=True, metrics=pack(current_metrics), baseline=pack(parent_metrics),
        delta=pack({k: current_metrics[k]-parent_metrics[k] for k in current_metrics}),
        matrix=[dict(stage=s, task=t, current=str(rates[s,t]), parent=str(base[s,t]))
                for s,t in rates],
        metric_gate=current_metrics["AUC"] > parent_metrics["AUC"],
        summed_epoch_seconds=sum(r["seconds"] for r in history)))
    write(O/"RUN_STATUS.json", dict(status="AWAITING_PROCESS_EXIT_ACCEPTANCE"))


if __name__ == "__main__":
    main()
