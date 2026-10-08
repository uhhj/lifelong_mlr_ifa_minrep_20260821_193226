"""Few real MLR+IFA batches, no baseline training run or rollout tuning."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from . import tests
from .control import Controller,Settings,rng_state,restore_rng
from .runtime import capture,restore
from .train import build
from .prepare import ROOT,OUT,config
from .io import save,write
from libero.lifelong.algos.mlr import MLR,build_dataset_for_task,update_buffer_equal_quota_10,build_ifa_similarity_matrix


def equal(a,b):
    import numpy as np
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful()
    dist.init_process_group('nccl',init_method='env://');torch.cuda.set_device(0)
    checks={};details={}
    try:
        with tempfile.TemporaryDirectory(prefix='e1_smoke_',dir=str(ROOT/'runs')) as d:
            cfg=EasyDict(config(42,d));cfg.e1_report=str(Path(d)/'reports')
            algo,b=build(cfg)
            # Isolated copy of precisely the buffer created by original task6 end logic.
            update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,seed=42,rate=.5)
            ds=build_dataset_for_task(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
            info=build_ifa_similarity_matrix(7,cfg.feature_dir,cfg.buffer_dir)
            algo.e1_stage=7;algo.start_task(-1,DataLoader(ds,batch_size=10));algo.policy.train()
            old=[i for i,p in enumerate(ds.file_paths) if Path(p).name.startswith('t6__')]
            new=[i for i,p in enumerate(ds.file_paths) if not Path(p).name.startswith('t')]
            ids=old[:5]+new[:5]
            data=lambda:default_collate([ds[i] for i in ids])
            # Two engineering steps initialize real AdamW moments and nonzero LR.
            for _ in range(2):MLR.observe(algo,data(),7,0,info)
            original=capture(algo)
            check_data=algo.map_tensor_to_device(default_collate([ds[i] for i in old[:20]])[0])
            def run(settings):
                c=Controller(algo.policy.module,42,settings)
                # every=1 only in engineering smoke, never persisted formal settings.
                return c.update(lambda:MLR.observe(algo,data(),7,0,info),lambda:algo._optimizer_step_count,
                    lambda rng,n:(check_data,[Path(ds.file_paths[i]).name for i in old[:20]]),dict(smoke=True))
            MLR.observe(algo,data(),7,0,info);native=capture(algo)
            restore(algo,original);run(Settings(enabled=False));checks['disabled_native_state_parity']=equal(native,capture(algo))
            restore(algo,original);run(Settings(every=1,rho=0.));checks['rho_zero_native_state_parity']=equal(native,capture(algo))
            c=Controller(algo.policy.module,42);state=capture(algo)
            a=c.losses(check_data);aa=c.losses(check_data)
            checks['deterministic_original_GMM_NLL']=torch.equal(a,aa)
            grads=[None if p.grad is None else p.grad.clone() for p in algo.policy.parameters()]
            c.losses(check_data,True)
            checks['check_gradient_isolation']=all((x is None and p.grad is None) or (x is not None and torch.equal(x,p.grad)) for x,p in zip(grads,algo.policy.parameters()))
            checks['check_modes_buffers_rng_unchanged']=equal(state,capture(algo))
            restore(algo,original);_,row=run(Settings(every=1));after=capture(algo)
            checks['one_optimizer_and_scheduler_step']=after['optimizer_step']==original['optimizer_step']+1 and after['scheduler_step']==original['scheduler_step']+1
            checks['moments_scheduler_rng_native_parity']=all(equal(native[k],after[k]) for k in ['optimizer','scheduler','rng','optimizer_step','scheduler_step'])
            checks['correction_global_cap']=row['correction_norm']<=.25*row['delta_norm']*(1+2e-6)
            checks['allowed_parameter_block']=all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in algo.policy.module.named_parameters() if p.requires_grad)
            checks['legal_check_buffer']=all(Path(ds.file_paths[i]).resolve().parent==Path(cfg.buffer_dir).resolve() for i in old[:20])
            f=Path(d)/'restore.pt';save(f,after)
            restore(algo,original)
            with torch.no_grad():next(algo.policy.parameters()).add_(1.)
            torch.rand(3,device='cuda')
            restore(algo,torch.load(f,map_location='cpu'));checks['epoch_state_save_load']=equal(after,capture(algo))
            from .legacy_metrics import metrics
            checks['legacy_metric_recomputation']=all(all(metrics(json.loads((ROOT/('reports/final_paper_text_faithful/primary_ep99/ifa_seed%d.json'%seed)).read_text())['r'])[k]==json.loads((ROOT/('reports/final_paper_text_faithful/primary_ep99/ifa_seed%d.json'%seed)).read_text())[k] for k in ['auc','nbt','fwt']) for seed in [42,43,44])
            from . import eval_metric
            from libero.lifelong.metric import evaluate_one_task_success as original_eval
            # The epoch loop copy differs only in resume start; native observe not overridden in base.
            import inspect,ast,textwrap
            from .baseline_loop import E1EpochLoop
            a=inspect.getsource(MLR.learn_all_tasks)
            z=inspect.getsource(E1EpochLoop.learn_all_tasks).replace('range(self.e1_start_epoch, self.cfg.train.n_epochs)','range(0, self.cfg.train.n_epochs)')
            checks['baseline_loop_unchanged_except_resume']=ast.dump(ast.parse(textwrap.dedent(a)))==ast.dump(ast.parse(textwrap.dedent(z)))
            observed=inspect.getsource(eval_metric.evaluate_one_task_success)
            observed='\n'.join(line for line in observed.splitlines() if 'record_trial(cfg,' not in line)
            checks['rollout_only_adds_trial_recording']=ast.dump(ast.parse(observed))==ast.dump(ast.parse(inspect.getsource(original_eval)))
            details=dict(real_smoke_row=row,trainable_count=sum(p.numel() for p in c.params),ifa_pairs=info['selected_pairs'],unit_tests=result.testsRun)
        write(OUT/'ENGINEERING_CHECK.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details,baseline_training_rerun=False))
        assert all(checks.values()),checks
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
