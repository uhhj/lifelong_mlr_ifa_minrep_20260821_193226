import torch
from torch import nn
from torch.nn import functional as F


class EffectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1543,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.output = nn.Linear(128,6)

    def forward(self, feature, action):
        # Temporal mean of exactly the original two visual tokens + one state token.
        if feature.ndim == 5:
            feature = feature.squeeze(1)
        assert feature.shape[1:] == (8,4,512)
        x = feature[:,:,1:,:].mean(1).reshape(len(feature),1536)
        h = self.body(torch.cat((x,action),-1))
        return self.output(h), F.normalize(h, dim=-1, eps=1e-8)
