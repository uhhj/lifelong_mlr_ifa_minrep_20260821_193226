import torch


def objective(policy,records,device):
    feature=torch.stack([r["feature"] for r in records]).to(device)
    action=torch.stack([r["action"] for r in records]).to(device)
    weight=torch.tensor([r["weight"] for r in records],device=device)
    z=policy.temporal_encode(feature)[:,-1]
    means,std,logits=policy.policy_head.forward_fn(z)
    gaussian=torch.distributions.Independent(torch.distributions.Normal(means,std),1)
    logp=gaussian.log_prob(action[:,None])
    recover=-torch.logsumexp(logits.log_softmax(-1)+logp,-1)
    execute=-torch.logsumexp((logits/.25).log_softmax(-1)+logp,-1)
    # Fixed per-window weights, denominator is the four-sample batch size.
    loss=.1*(weight*(recover+.5*execute)).mean()
    assert torch.isfinite(loss)
    return loss,dict(recover=float((weight*recover).mean().detach()),
                     execution=float((weight*execute).mean().detach()))
