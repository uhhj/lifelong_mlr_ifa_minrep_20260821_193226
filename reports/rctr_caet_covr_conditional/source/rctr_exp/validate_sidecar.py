"""Backward-only admission on actual certified records; never optimizer updates."""
import torch
from coreset_exp.common import O, rng, restore_rng, sha
from coreset_exp.model import cfg, strict_downstream
from libero.lifelong.models.brief import BriefPolicy
from .loss import objective


def validate(path, checkpoint):
    records = torch.load(path, map_location="cpu")
    if not records:
        return dict(status="NO_CERTIFIED_RECORDS", optimizer_calls=0, checked=0)
    state = rng()
    try:
        policy = BriefPolicy(cfg("ra_coreset"), None).cuda().eval()
        strict_downstream(policy, checkpoint)
        policy.requires_grad_(True)
        scale = torch.load(O/"pretraining_action_scale.pth").cuda()
        results = []
        for record in records:
            policy.zero_grad()
            assert record["certified"]
            for field in ("zdelta", "a_plus", "a_minus"):
                assert torch.isfinite(record[field]).all()
            assert record["zdelta"].shape == (8, 4, 512)
            if record["has_normal_reference"]:
                assert record["z0"].shape == (8, 4, 512)
            loss, parts = objective(policy, record, scale, torch.device("cuda"))
            loss.backward()
            grads = [p.grad for p in policy.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norm = sum(float(g.abs().sum()) for g in grads)
            assert norm > 0, "Real certified supervision has no reachable gradient"
            results.append(dict(initial=record["initial"],
                snapshot_step=record["snapshot_step"], gradient_l1=norm,
                positive_gripper_sign=int(torch.sign(record["a_plus"][-1])),
                negative_gripper_sign=int(torch.sign(record["a_minus"][-1])), **parts))
        return dict(status="PASS", optimizer_calls=0, checked=len(results),
                    checkpoint_sha256=sha(checkpoint), records=results)
    finally:
        restore_rng(state)
