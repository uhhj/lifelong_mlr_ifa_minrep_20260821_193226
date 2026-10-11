import copy
import random
import numpy as np
import torch

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])


def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)


def capture(algo):
    return dict(model=cpu(algo.policy.state_dict()), optimizer=cpu(algo.optimizer.state_dict()),
                scheduler=copy.deepcopy(algo.scheduler.state_dict()) if algo.scheduler else None,
                optimizer_step=algo._optimizer_step_count, scheduler_step=algo._scheduler_step_count,
                rng=cpu(rng_state()), modes={n:m.training for n,m in algo.policy.named_modules()})


def capture_device(algo):
    """Bounded device-side pre-update snapshot for nonfinite fail-stop evidence.

    No host transfer or RNG draw on the successful path. This extra memory/copy
    cost is included in measured training time and peak memory.
    """
    return dict(model=copy.deepcopy(algo.policy.state_dict()),
                optimizer=copy.deepcopy(algo.optimizer.state_dict()),
                scheduler=copy.deepcopy(algo.scheduler.state_dict()) if algo.scheduler else None,
                optimizer_step=algo._optimizer_step_count,scheduler_step=algo._scheduler_step_count,
                rng=cpu(rng_state()),modes={n:m.training for n,m in algo.policy.named_modules()})


def restore(algo, state):
    algo.policy.load_state_dict(state['model'], strict=True)
    algo.optimizer.load_state_dict(state['optimizer'])
    if algo.scheduler: algo.scheduler.load_state_dict(state['scheduler'])
    algo._optimizer_step_count=state['optimizer_step']; algo._scheduler_step_count=state['scheduler_step']
    for n,m in algo.policy.named_modules(): m.training=state['modes'][n]
    restore_rng(state['rng'])


