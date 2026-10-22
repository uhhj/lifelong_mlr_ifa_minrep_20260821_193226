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
from .variant import VARIANT,METHOD
from .model import Student,DeployResidual,ResidualHead
from coreset_exp.common import rng,restore_rng


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
    assert sum(r["batches"] for r in history) == history[-1]["steps"] == 105500
    if VARIANT=="EFTR":assert js(O/"TASK6_TENSOR_PARITY.json")["equal"]
    assert js(root/"INITIAL_STATE.json") == js(parent/"ra_coreset/INITIAL_STATE.json")
    assert js(O/"GPU_PREFLIGHT.json")["status"] == "PASS"
    policy = BriefPolicy(cfg("ra_coreset"), None).cpu()
    checkpoints, selections, ifa = [], {}, {}
    deployments=[]
    for task in range(6, 10):
        cp = root/"models"/("mlrifa_%d_model_ep99.pth" % task)
        raw = torch.load(cp, map_location="cpu")
        assert raw["task"] == task and raw["epoch_index"] == 99
        assert raw["contract_sha256"] == sha(O/"CONTRACT.json")
        checkpoints.append(dict(task=task, sha256=sha(cp),
            strict_loaded_keys=strict_downstream(policy, cp)))
        branch=root/'deployed_residual'/('task%d.pth'%task);meta=js(branch.with_suffix('.json'));aux=root/'aux_heads'/('task%d.pth'%task)
        assert meta['sha256']==sha(branch) and meta['main_checkpoint_sha256']==sha(cp) and meta['training_student_sha256']==sha(aux)
        deploy=DeployResidual();dstate=torch.load(branch,map_location='cpu');assert len(dstate)==14;deploy.load_state_dict(dstate,strict=True)
        training=Student();training.load_state_dict(torch.load(aux,map_location='cpu')['model'],strict=True)
        assert all(torch.equal(dstate[k],training.deploy.state_dict()[k]) for k in dstate)
        h=torch.linspace(-1,1,3*512).reshape(3,512);policy.eval();deploy.eval();training.eval()
        with torch.no_grad():
            direct=policy.policy_head.forward_fn(training.deploy(h)[0]);wrapped=ResidualHead(policy.policy_head,deploy).forward_fn(h)
            assert all(torch.equal(x,y) for x,y in zip(direct,wrapped))
        deployments.append(dict(task=task,sha256=sha(branch),strict_loaded_keys=14,aux_sha256=sha(aux),student_keys=20,training_deployment_exact=True))
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
    from .audit_mechanism import audit as effect_audit
    auxiliary = effect_audit()
    seen_steps=[]
    for row in history:
        stats=js(root/"auxiliary"/("task%d_epoch%02d.json"%(row["task"],row["epoch"])))
        assert len(stats)==row["batches"]
        seen_steps.extend(x["step"] for x in stats)
        assert all(x["committed_updates"]==1 for x in stats)
    assert seen_steps==list(range(105500))
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
    for row in new.values():
        assert row['loaded_residual_keys']==14 and row['deployment_residual_sha256']==deployments[row['stage']-6]['sha256']
    from coreset_exp import evaluate as unused_context

    current_metrics, parent_metrics = metrics(rates), metrics(base)
    verify()
    write(O/"AUDIT.json", dict(status="PASS", method=METHOD, epochs=400,
        optimizer_steps=105500, formal_records=200, vector_attempts=20,
        checkpoints=checkpoints, deployments=deployments, selection=selections, ifa=ifa, auxiliary=auxiliary,
        all_inputs_unchanged=True, metrics=pack(current_metrics), baseline=pack(parent_metrics),
        delta=pack({k: current_metrics[k]-parent_metrics[k] for k in current_metrics}),
        matrix=[dict(stage=s, task=t, current=str(rates[s,t]), parent=str(base[s,t]))
                for s,t in rates],
        metric_gate=current_metrics["AUC"] > parent_metrics["AUC"],
        summed_epoch_seconds=sum(r["seconds"] for r in history)))
    write(O/"RUN_STATUS.json", dict(status="AWAITING_PROCESS_EXIT_ACCEPTANCE"))


if __name__ == "__main__":
    main()
