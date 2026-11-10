"""Full-S update backbone stripped to original/ALIGN BC + 0.1 IFA.

The order of zero_grad, hidden, masked_bc, IFA, loss_scale, backward,
finite checks, gradient clipping, optimizer and scheduler is inherited from S.
No execution expert, retention, support weight, candidate, gate, probe or
collection path is imported or reachable from this module.
"""
from case_runtime import *


def update(a, batch, info, stage):
    variant = INPUT['variant']
    assert variant in ('ALIGN', 'MLR')
    x = a.map_tensor_to_device(batch[0])
    tasks = torch.tensor([int(v) for v in batch[1]], device='cuda')
    a.optimizer.zero_grad()
    h = hidden(a.policy.module, x['feature'])
    bc, valid = masked_bc(a.policy.module, h, x, variant == 'ALIGN')
    mask = tasks == stage
    olds = [a.refs[j if k == stage else k] for j, k in info['selected_pairs']]
    ifa = angular_mean(h[mask], a.refs[stage], olds, a.cfg.gamma) if mask.any() else h.sum() * 0
    loss = bc + .1 * ifa
    (a.loss_scale * loss).backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
    torch.nn.utils.clip_grad_norm_(a.policy.parameters(), a.cfg.train.grad_clip)
    a.optimizer.step()
    a.scheduler.step()
    a.step += 1
    return dict(loss=float(loss.detach()), bc=float(bc.detach()), ifa=float(ifa.detach()),
                valid=int(valid.sum()), invalid=int((~valid).sum()))
