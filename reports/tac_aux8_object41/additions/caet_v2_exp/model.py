import torch
from torch import nn
from torch.nn import functional as F
from .variant import VARIANT


class EffectModel(nn.Module):
    def __init__(self):
        super().__init__()
        if VARIANT=='RSET':
            self.project=nn.Linear(1536,128)
            self.temporal=nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True)
            self.pool=nn.Linear(128,1)
            pos=torch.arange(8).float()[:,None];freq=torch.exp(torch.arange(0,128,2).float()*(-__import__('math').log(10000)/128))
            pe=torch.zeros(8,128);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
            self.register_buffer('positions',pe)
        self.body = nn.Sequential(nn.Linear(135 if VARIANT=='RSET' else 1543,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.output = nn.Linear(128,7)

    def forward(self, feature, action):
        # Temporal mean of exactly the original two visual tokens + one state token.
        if feature.ndim == 5:
            feature = feature.squeeze(1)
        assert feature.shape[1:] == (8,4,512)
        if VARIANT=='RSET':
            z=self.project(feature[:,:,1:,:].reshape(len(feature),8,1536))+self.positions[None]
            z=self.temporal(z);x=(z*self.pool(z).softmax(1)).sum(1)
        else:x = feature[:,:,1:,:].mean(1).reshape(len(feature),1536)
        h = self.body(torch.cat((x,action),-1))
        return self.output(h), F.normalize(h, dim=-1, eps=1e-8)
