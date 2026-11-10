"""Count actual calls without changing updates; no counter rollback on restore."""
from contextlib import contextmanager
CALLS={'production':0,'admission':0}
AUTOGRAD={'production':0,'admission':0}
MODE='production'
_installed=False
def install():
    global _installed
    if _installed:return
    import torch
    original=torch.optim.AdamW.step
    grad=torch.autograd.grad
    def step(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        CALLS[MODE]+=1
        return value
    def autograd(*args,**kwargs):
        value=grad(*args,**kwargs)
        AUTOGRAD[MODE]+=1
        return value
    torch.optim.AdamW.step=step;torch.autograd.grad=autograd;_installed=True
@contextmanager
def admission():
    global MODE
    previous=MODE;MODE='admission'
    try:yield
    finally:MODE=previous
