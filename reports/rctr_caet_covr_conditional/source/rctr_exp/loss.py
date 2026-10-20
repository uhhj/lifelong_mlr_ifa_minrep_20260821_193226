"""RCTR response/route/preference objective; no action-distribution replacement."""
import torch
import torch.nn.functional as F


def parameters(policy, feature):
    last = policy.temporal_encode(feature)[:, -1]
    return policy.policy_head.forward_fn(last)


def nearest(means, target, action_scale):
    # Six continuous coordinates use fixed pretraining scale; gripper uses raw units.
    d = torch.cat([action_scale[:6], action_scale.new_ones(1)])
    return (((means.detach()-target[:, None])/d)**2).sum(-1).argmin(-1)


def objective(policy, record, action_scale, device):
    xd = record["zdelta"].to(device)[None]
    ap = record["a_plus"].to(device)[None]
    an = record["a_minus"].to(device)[None]
    means, std, logits = parameters(policy, xd)
    assert means.shape == std.shape == (1, 5, 7)
    j = nearest(means, ap, action_scale)
    batch = torch.arange(len(j), device=device)
    selected_logit = logits[batch, j]
    other = logits.masked_fill(F.one_hot(j, logits.shape[-1]).bool(), -torch.inf).max(-1).values
    route = F.relu(.1 + other-selected_logit).mean()
    winner = logits.detach().argmax(-1)
    mu = means[batch, winner]
    sigma = std[batch, winner]
    assert (sigma > 0).all()
    # Standardized squared distance under the actually selected Gaussian; no logit
    # gradient through argmax. Route loss separately trains its execution priority.
    sp = -(((ap-mu)/sigma)**2).mean(-1)
    sn = -(((an-mu)/sigma)**2).mean(-1)
    pref = F.softplus(.1+sn-sp).mean()
    response = means.sum()*0
    if record["has_normal_reference"]:
        x0 = record["z0"].to(device)[None]
        a0 = record["a0"].to(device)[None]
        m0, _, _ = parameters(policy, x0)
        j0 = nearest(m0, a0, action_scale)
        delta = (means[batch, j]-m0[batch, j0])[:, :6]
        target = (ap-a0)[:, :6]
        # Mathematical squared vector norm, then average over records.
        response = (((delta-target)/action_scale[:6])**2).sum(-1).mean()
    total = .1*response + .02*route + .02*pref
    assert torch.isfinite(total)
    return total, dict(response=float(response.detach()), route=float(route.detach()),
                      preference=float(pref.detach()), route_component=int(j[0]),
                      winner_component=int(winner[0]))
