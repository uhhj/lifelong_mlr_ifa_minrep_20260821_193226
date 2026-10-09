import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from easydict import EasyDict
from .runtime import SelectionMLR
from .io import sha,write
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import control_seed,get_task_embs

O=Path(os.environ['E1_OUT']);S=Path(os.environ['E1_SOURCE'])

def build(cfg):
    assert torch.cuda.is_available()
    control_seed(cfg.seed)
    benchmark=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    benchmark.set_task_embs(get_task_embs(cfg,[benchmark.get_task(i).language for i in range(benchmark.n_tasks)]))
    Path(cfg.experiment_dir).mkdir(parents=True,exist_ok=True);Path(cfg.buffer_dir).mkdir(parents=True,exist_ok=True)
    algo=SelectionMLR(1,cfg).cuda();algo.policy=DDP(algo.policy,device_ids=[0],output_device=0)
    return algo,benchmark

def main():
    assert json.loads((O/'REAL_PREFLIGHT.json').read_text())['status']=='PASS'
    contract=json.loads((O/'CONTRACT.json').read_text())
    assert all(sha(S/k)==v for k,v in contract['source_sha256'].items())
    cfg=EasyDict(json.loads((O/'resolved_training.json').read_text()))
    done=O/'TRAINING_COMPLETE.json'
    if done.exists():return
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats();begin=time.time()
    try:
        algo,benchmark=build(cfg)
        write(Path(cfg.experiment_dir)/'config.json',dict(cfg));algo.train()
        for task in range(6,10):algo.learn_all_tasks(None,benchmark,{},task)
        checkpoints=sorted(Path(cfg.experiment_dir).glob('mlrifa_*_model_ep*.pth'));assert len(checkpoints)==28
        write(done,dict(status='TRAINED_EVAL_PENDING',seed=42,epochs=400,invocation_seconds=time.time()-begin,
          peak_gpu_bytes=torch.cuda.max_memory_allocated(),selection_totals=algo.selection_totals,
          checkpoints={p.name:sha(p) for p in checkpoints}))
        write(O/'RUN_STATUS.json',dict(status='TRAINED_EVAL_PENDING',updated_unix=time.time()))
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
