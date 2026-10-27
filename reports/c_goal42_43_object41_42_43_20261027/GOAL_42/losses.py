"""C/R/G only: unchanged C, V2-J margin control, uploaded directional G."""
import torch
import torch.nn.functional as F
from legacy_losses import cost, w2, route
from directional_prototype import auxiliary_v3


def margin_j(logits, score):
    # Preserve V2's highest-logit tie reference; only replace route functional form.
    tied = score.detach() == score.detach().min(-1, keepdim=True).values
    ref = logits.detach().masked_fill(~tied, float('-inf')).argmax(-1)
    return route(logits, F.one_hot(ref, logits.shape[-1]).bool())


def reference_r(q, y, teacher, scale, weights):
    """Independent R expression used only by isolated real-update admission."""
    mu, sd, z = q
    tm, ts, tz = (x.detach() for x in teacher)
    i = torch.arange(len(y), device=y.device)
    tk = tz.argmax(-1)
    c = cost(mu, sd, y.detach(), scale.detach())
    d = w2(mu, sd, tm[i,tk], ts[i,tk], scale.detach())
    score = (c + weights.detach()[:,None]*d)/(1 + weights.detach()[:,None])
    ties = score.detach() == score.detach().min(-1,keepdim=True).values
    ref = z.detach().masked_fill(~ties, float('-inf')).argmax(-1)
    one = F.one_hot(ref, z.shape[1]).bool()
    hinge = F.relu(.1 + z.masked_fill(one, -1e9).max(-1).values - z[i,ref])
    return (score[i,z.detach().argmax(-1)] + .1*hinge).mean()


def aux(arm, q, y, teacher, scale, weights, anchor_strength=1.):
    assert arm in 'CRG'
    mu, sd, z = q
    t = tuple(x.detach() for x in teacher)
    i = torch.arange(len(y), device=y.device); k = z.detach().argmax(-1)
    if arm == 'G':
        value, stats, terms = auxiliary_v3(q, y, t, scale, weights,
                                         anchor_strength=anchor_strength)
        return value, stats, terms['expert'], terms['route'], terms['anchor']
    c = cost(mu, sd, y.detach(), scale.detach())
    if arm == 'C':
        score = c
        rr = route(z, F.one_hot(c.detach().argmin(-1), z.shape[1]).bool())
    else:
        tm, ts, tz = t; tk = tz.argmax(-1)
        d = w2(mu, sd, tm[i,tk], ts[i,tk], scale.detach())
        w = weights.detach()[:,None]
        score = (c+w*d)/(1+w)
        rr = margin_j(z, score)
    ex = score[i,k]
    value = (ex+.1*rr).mean()
    stats = dict(count=len(y), expert_selected_sum=float(c[i,k].detach().sum()),
                 exec_score_sum=float(ex.detach().sum()), route_sum=float(rr.detach().sum()),
                 expert_regret_sum=float((c[i,k]-c.min(-1).values).detach().sum()))
    return value, stats, ex.mean(), rr.mean(), ex.sum()*0
