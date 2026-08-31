#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "paper_reproduction_v2" / "evaluation_audit.json"


def main():
    metric = (ROOT / "libero" / "lifelong" / "metric.py").read_text(encoding="utf-8")
    policy = (ROOT / "libero" / "lifelong" / "models" / "complete_policy.py").read_text(encoding="utf-8")
    head = (ROOT / "libero" / "lifelong" / "models" / "policy_head.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "libero" / "lifelong" / "test.py").read_text(encoding="utf-8")
    checks = {
        "twenty_trials_runner": "eval.n_eval=20" in (ROOT / "scripts" / "eval_object_task6_reconciliation.sh").read_text(),
        "saved_init_states": "torch.load(init_states_path)" in metric,
        "first_twenty_states_in_two_ten_env_groups": "env_num =  10" in metric and "np.arange(i * env_num" in metric,
        "environment_seed_propagation": metric.count("env.seed(seed)") >= 2 and "seed = see + i" in metric,
        "environment_reset": "env.reset()" in metric,
        "policy_reset_each_rollout_group": "algo.reset()" in metric,
        "success_from_done": "dones[k] = dones[k] or done[k]" in metric,
        "ddp_prefix_stripped_on_checkpoint_load": "OrderedDict((k[7:], v)" in evaluator,
        "highest_logit_component": "max_idx = logits.argmax(dim=-1)" in head,
        "stochastic_selected_normal_sample": "dist.sample().detach().cpu()" in policy,
        "formal_checkpoint_grid": "[70,75,80,85,90,95]" in (ROOT / "scripts" / "eval_object.sh").read_text(),
    }
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "trials_per_cell": 20,
        "init_state_source": "bundled LIBERO task init_states_file",
        "action_protocol": "highest-logit GMM component, then sample selected Normal",
        "checks": checks,
        "pass": all(checks.values()),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
