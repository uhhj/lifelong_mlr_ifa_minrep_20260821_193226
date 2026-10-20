"""Shared training/native selected-Gaussian correction and numerical KL enforcement."""
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

def calibrate_distribution(b, scales, logits, delta, epsilon=0.5):
    assert b.ndim == 3 and b.shape[1:] == (5, 7)
    assert scales.shape == b.shape and logits.shape == b.shape[:2] and delta.shape == (len(b), 7)
    assert b.dtype in (torch.float32, torch.float64)
    assert all(torch.isfinite(x).all() for x in (b, scales, logits, delta))
    assert (scales > 0).all()
    rows = torch.arange(len(b), device=b.device)
    selected = logits.argmax(-1)
    base_means = b.tanh()
    base = base_means[rows, selected]
    sigma = scales[rows, selected]
    proposal = (b[rows, selected] + delta).tanh()
    shift = proposal.double() - base.double()
    raw_kl = 0.5 * ((shift / sigma.double()) ** 2).sum(-1)
    alpha = torch.ones_like(raw_kl)
    if epsilon is not None:
        assert epsilon == 0.5, "Frozen scientific budget"
        alpha = torch.minimum(alpha, (epsilon / raw_kl.clamp_min(1e-30)).sqrt())
    executed = (base.double() + alpha[:, None] * shift).to(b.dtype)
    # The value passed to Normal, not a pre-cast surrogate, must satisfy the bound.
    # nextafter only corrects floating-point boundary overshoot toward the parent.
    # Preserve analytical projection gradients; discrete ULP rounding is not a learned gate.
    rounding_passes = 0
    if epsilon is not None:
        for _ in range(16):
            actual = 0.5 * (((executed.double() - base.double()) / sigma.double()) ** 2).sum(-1)
            over = actual.detach() > epsilon
            if not bool(over.any()):
                break
            toward = torch.nextafter(executed.detach(), base.detach())
            rounded = torch.where(over[:, None], toward, executed.detach())
            executed = executed + (rounded - executed).detach()
            rounding_passes += 1
    actual = 0.5 * (((executed.double() - base.double()) / sigma.double()) ** 2).sum(-1)
    if epsilon is not None:
        assert bool((actual.detach() <= epsilon + 1e-12).all()), "Actual executed KL exceeds budget"
    means = base_means.clone()
    means[rows, selected] = executed
    dist = Independent(Normal(executed, sigma), 1)
    return dict(means=means, scales=scales, logits=logits, selected=selected,
                probs=logits.softmax(-1), base_means=base_means, base_selected=base,
                executed_mean=executed, executed_scale=sigma, executed_dist=dist,
                raw_kl=raw_kl, actual_kl=actual, alpha=alpha, rounding_passes=rounding_passes,
                gmm=MixtureSameFamily(Categorical(logits=logits), Independent(Normal(means, scales), 1)))
