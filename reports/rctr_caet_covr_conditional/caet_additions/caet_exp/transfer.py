import torch


def pair_masks(effect, moving, tasks):
    distance = torch.cdist(effect,effect,p=2)
    different = tasks[:,None] != tasks[None,:]
    active = moving[:,None] & moving[None,:]
    positive = different & active & (distance <= .5)
    negative = different & (distance >= 1.)
    return positive, negative


def transfer_loss(embedding, effect, moving, tasks):
    positive, negative = pair_masks(effect,moving,tasks)
    eligible = positive.any(-1) & negative.any(-1)
    if not eligible.any():
        return embedding.sum()*0, 0
    logits = embedding @ embedding.T / .1
    selected = (positive | negative)[eligible]
    scores = logits[eligible]
    denominator = torch.logsumexp(scores.masked_fill(~selected,-torch.inf),-1)
    log_prob = scores-denominator[:,None]
    pos = positive[eligible]
    loss = -(log_prob.masked_fill(~pos,0).sum(-1)/pos.sum(-1)).mean()
    return loss, int(pos.sum())
