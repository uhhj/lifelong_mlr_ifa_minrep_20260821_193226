import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP

from .runtime import P1MLR
from .io import write, sha
from .prepare import ROOT, OUT, config
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import control_seed, get_task_embs, safe_device


def build(cfg):
    control_seed(cfg.seed)
    benchmark=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions=[benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    benchmark.set_task_embs(get_task_embs(cfg,descriptions))
    Path(cfg.experiment_dir).mkdir(parents=True,exist_ok=True)
    Path(cfg.buffer_dir).mkdir(parents=True,exist_ok=True)
    algo=safe_device(P1MLR(1,cfg),cfg.device)
    algo.policy=DDP(algo.policy,device_ids=[0],output_device=0)
    return algo,benchmark


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    assert a.seed in (42,43,44)
    assert json.loads((OUT/'ENGINEERING_CHECK.json').read_text())['status']=='PASS'
    if a.seed!=42: assert json.loads((OUT/'SEED42_DECISION.json').read_text())['run_additional_seeds']
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protocol['implementation_files'].items()),'P1_CODE_FREEZE_MISMATCH'
    cfg=EasyDict(json.loads((OUT/('CONFIG_seed%d.json'%a.seed)).read_text()))
    done=Path(cfg.p1_root)/'TRAINING_COMPLETE.json'
    if done.exists():return
    if not dist.is_initialized():dist.init_process_group('nccl',init_method='env://')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    begin=time.monotonic()
    try:
        algo,benchmark=build(cfg)
        write(Path(cfg.experiment_dir)/'config.json',dict(cfg))
        algo.train()
        for task in range(cfg.start_task,cfg.end_task):
            algo.learn_all_tasks(None,benchmark,{},task)
        expected=[Path(cfg.experiment_dir)/('mlrifa_%d_model_ep99.pth'%t) for t in range(6,10)]
        assert all(p.is_file() for p in expected)
        write(done,dict(status='PASS',seed=a.seed,epochs_per_task=100,tasks=[6,7,8,9],
            invocation_seconds=time.monotonic()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            totals=algo.p1_totals,checkpoints=[str(p) for p in expected]))
        write(Path(cfg.p1_report)/'STATUS.json',dict(status='TRAINING_COMPLETE_EVALUATION_PENDING',seed=a.seed,
            p1_training_active=False,baseline_training_rerun=False))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
