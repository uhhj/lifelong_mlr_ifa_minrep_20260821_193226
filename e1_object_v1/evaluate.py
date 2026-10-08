"""Invoke original evaluation, with observation-only per-trial records."""
import json
import os
import re
from pathlib import Path

CURRENT={}


def record_trial(cfg,task,seed,trial,initial_state,success,steps):
    assert CURRENT, 'Evaluation checkpoint provenance missing'
    path=Path(os.environ['E1_TRIAL_LOG']);path.parent.mkdir(parents=True,exist_ok=True)
    row=dict(CURRENT,seed=seed,task=task,trial=trial,initial_state_index=initial_state,
             success=success,vector_group_steps=steps)
    with path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()


def main():
    from .bindings import bind
    bind()
    import torch
    import libero.lifelong.test as baseline
    import libero.lifelong.metric as metric
    from . import eval_metric
    # Import canonical module so record_trial and loader share the same context under -m.
    from e1_object_v1 import evaluate as context
    native_load=torch.load
    def observed_load(path,*args,**kwargs):
        result=native_load(path,*args,**kwargs)
        if isinstance(path,(str,Path)):
            m=re.fullmatch(r'mlrifa_(\d+)_model_ep(\d+)\.pth',Path(path).name)
            if m:context.CURRENT.update(checkpoint=str(path),stage=int(m[1]),epoch=int(m[2]))
        return result
    torch.load=observed_load
    metric.evaluate_one_task_success=eval_metric.evaluate_one_task_success
    baseline.main()


if __name__=='__main__':main()
