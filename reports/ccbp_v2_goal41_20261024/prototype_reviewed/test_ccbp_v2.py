"""CPU synthetic checks only; no real robot checkpoints or data are used."""
import json
from pathlib import Path

import torch
from losses_ccbp_v2 import auxiliary_v2, estimate_teacher_cost_median, regret_route


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    dt = torch.float64
    n, k = 3, 5
    y = torch.randn(n, 7, dtype=dt) * 0.1
    scale = torch.tensor([0.46725568, 0.1, 1.0], dtype=dt)
    tm = torch.randn(n, k, 7, dtype=dt) * 0.2
    ts = torch.full_like(tm, 1e-4)
    tl = torch.tensor([[3., 1., 0., -1., -2.]] * n, dtype=dt)
    teacher = (tm.requires_grad_(), ts.requires_grad_(), tl.requires_grad_())
    tau = estimate_teacher_cost_median(teacher, y, scale)
    mu = (torch.randn(n, k, 7, dtype=dt) * 0.2).requires_grad_()
    sd = torch.full_like(mu, 1e-4, requires_grad=True)
    logits = torch.tensor([[0., 3., 1., -1., -2.]] * n, dtype=dt, requires_grad=True)
    args = dict(teacher_cost_median=tau)
    loss, _ = auxiliary_v2((mu, sd, logits), y, teacher, scale, **args)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (mu, sd, logits))
    assert all(t.grad is None for t in teacher)
    perm = torch.tensor([3, 0, 4, 1, 2])
    moved, _ = auxiliary_v2(
        (mu[:, perm], sd[:, perm], logits[:, perm]), y,
        tuple(t[:, perm] for t in teacher), scale, **args
    )
    torch.testing.assert_close(loss, moved, rtol=1e-12, atol=1e-12)
    perm_error = float((loss.detach() - moved.detach()).abs())

    # A low-error window still receives a mean gradient: no c_teacher + epsilon dead zone.
    ey = torch.zeros(1, 7, dtype=dt)
    em = torch.zeros(1, 5, 7, dtype=dt)
    em[:, :, 0] = 0.01
    em.requires_grad_()
    es = torch.full_like(em, 1e-4, requires_grad=True)
    el = torch.tensor([[3., 2., 1., 0., -1.]], dtype=dt, requires_grad=True)
    et = (torch.zeros_like(em), torch.full_like(em, 1e-4), el.detach().clone())
    small, _ = auxiliary_v2((em, es, el), ey, et, torch.ones(3, dtype=dt),
                             teacher_cost_median=0.003, anchor_mode="quality")
    small.backward()
    assert em.grad[0, 0, 0] > 0

    def wrong_mode_gradient(bad_cost):
        zz = torch.tensor([[0., 2., -1., -2., -3.]], dtype=dt, requires_grad=True)
        costs = torch.tensor([[0.001, bad_cost, 0.002, 0.003, 0.004]], dtype=dt)
        rr = regret_route(zz, costs).mean()
        rr.backward()
        assert zz.grad[0, 0] < 0 and zz.grad[0, 1] > 0
        return float(zz.grad[0, 1])
    mild = wrong_mode_gradient(0.011)
    severe = wrong_mode_gradient(4.001)
    assert severe > mild

    # Teacher demonstration disagreement yields a lower frozen anchor weight.
    fixed_tau = 0.00386293618703643
    low_error = 0.00386293618703643
    high_error = 4.4274847785089655
    w_good = fixed_tau / (fixed_tau + low_error)
    w_bad = fixed_tau / (fixed_tau + high_error)
    assert 0 < w_bad < w_good < 1
    cost_tie = torch.tensor([[0.1, 0.1, 0.3, 0.4, 0.5]], dtype=dt)
    tie_logits = torch.tensor([[0., 1., 2., 3., 4.]], dtype=dt)
    torch.testing.assert_close(regret_route(tie_logits, cost_tie),
                               regret_route(tie_logits[:, perm], cost_tie[:, perm]))
    tied_logits = torch.zeros(1, 5, dtype=dt, requires_grad=True)
    tied = regret_route(tied_logits, torch.ones_like(tied_logits))
    assert tied.item() == 0

    result = {
        "scope": "synthetic_CPU_only_no_real_checkpoint_no_environment",
        "finite_nonzero_student_gradients": True,
        "teacher_has_no_gradient": True,
        "joint_permutation_loss_difference": perm_error,
        "low_cost_selected_mean_gradient": float(em.grad[0, 0, 0]),
        "mild_wrong_logit_gradient": mild,
        "severe_wrong_logit_gradient": severe,
        "gradient_ratio_same_logits": severe / mild,
        "teacher_quality_weight_at_training_median": w_good,
        "teacher_quality_weight_at_example_high_cost": w_bad,
        "equal_scores_have_zero_routing_penalty": True,
        "cost_tie_routing_is_permutation_invariant": True,
        "no_robot_training_or_rollouts": True,
    }
    Path(__file__).with_name("synthetic_test_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
