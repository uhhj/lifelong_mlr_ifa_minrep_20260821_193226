"""Real native entry/checkpoint boundary parity with zero robot rollouts."""
import os
import runpy
import random
import json
import numpy as np
import torch
from confirmation_experiment.common import ROOT,read,write,rng_state,state_digest
from confirmation_experiment.recovery.common import AUDIT,CELLS

def normalize_config(cfg):
    x=json.loads(json.dumps(cfg))
    for k in ('experiment_root','experiment_dir','experiment_name','test_log_file'):x.pop(k,None)
    return x

def main():
    import libero.lifelong.metric as metric
    mode=os.environ['RECOVERY_PROBE_MODE'];records=[]
    def probe(**kw):
        index=len(records);stage,task=CELLS[index]
        assert kw['task_ids']==[task]
        records.append({'cell':[stage,task],'model_sha256':state_digest(kw['algo'].policy.state_dict()),
             'rng_sha256':state_digest(rng_state()),'config_sha256':state_digest(normalize_config(kw['cfg'])),
             'backend':[torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic,torch.is_grad_enabled()],
             'module_modes':{n:m.training for n,m in kw['algo'].policy.named_modules()},'see':kw['see']})
        if not (mode=='reuse_first' and index==0):
            # Deliberately different first-cell RNG consumption tests whether
            # the native stage reset isolates the lost post-task6 RNG state.
            for _ in range(19):random.random()
            np.random.rand(31);torch.rand(37);torch.rand(41,device='cuda')
        return np.array([.25])
    metric.evaluate_success=probe
    runpy.run_path(str(ROOT/'libero/lifelong/test.py'),run_name='__main__')
    assert len(records)==10
    write(AUDIT/('EVAL_PROBE_'+mode+'.json'),{'mode':mode,'records':records,'robot_trials':0})

if __name__=='__main__':main()
