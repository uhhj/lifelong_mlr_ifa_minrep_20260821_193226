"""Unactivated candidate: independent snapshots on each tensor's existing device."""
import copy
import torch

def device_snapshot(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x,dict):return {k:device_snapshot(v) for k,v in x.items()}
    if isinstance(x,list):return [device_snapshot(v) for v in x]
    if isinstance(x,tuple):return tuple(device_snapshot(v) for v in x)
    return copy.deepcopy(x)
