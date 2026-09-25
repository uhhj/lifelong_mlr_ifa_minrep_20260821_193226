"""Independent source priors; native BC update, fresh AdamW, no replay/QP."""
import argparse
import contextlib
import io
import math
import os
import shutil
import time
import numpy as np
from .common import ROOT, OUT, STORE, read, write, freeze, sha, stable_seed


def main(seed, task):
    from cb_ifa_experiment import gpu_runtime as env
    from gradient_relation_experiment import kernel as g
    import torch
    from torch.utils.data.distributed import DistributedSampler
    assert read(OUT/('CACHE_COMPLETE_s%d_t%d.json'%(seed,task)))['status']=='PASS'
    proto = read(OUT/'TRAINING_PROTOCOL.json')
    run = STORE/'sources'/('s%d_t%d'%(seed,task))
    report = OUT/'sources'/('s%d_t%d'%(seed,task))
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    if (report/'COMPLETE.json').exists():
        return
    rows = read(OUT/'DATA_MANIFEST.json')['tasks'][str(task)]['windows']['A_train']
    feature, actions = [], []
    for path in sorted((STORE/'features'/('seed%d'%seed)/('task%d'%task)/'A_train').glob('chunk*.npz')):
        assert sha(path)==read(path.with_suffix('.json'))['file_sha256']
        with np.load(path) as z:
            feature.append(z['feature']); actions.append(z['actions'])
    feature = torch.from_numpy(np.concatenate(feature))
    actions = torch.from_numpy(np.concatenate(actions))
    assert len(feature)==len(rows)
    class Windows(torch.utils.data.Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            return dict(feature=feature[i:i+1],actions=actions[i:i+1]), str(task)
    ds=Windows();batch_size=proto['batch_size'];bpe=math.ceil(len(ds)/batch_size)
    sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=seed,drop_last=False)
    stream=[]
    for epoch in range(math.ceil(1000/bpe)):
        sampler.set_epoch(epoch);indices=list(sampler)
        stream.extend(dict(epoch=epoch,indices=indices[b:b+batch_size]) for b in range(0,len(indices),batch_size))
    stream=stream[:1000]
    freeze(report/'STREAM.json',dict(seed=seed,target=task,batches=stream,
           policy_seed=[stable_seed('task_relation_forecast_v1','train',seed,task,k) for k in range(1000)],
           learning_rates=proto['step_lr'][:1000]))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env.init_ddp(run/('ddp_%d'%os.getpid()))
    try:
        cfg=env.train_config(seed,run)
        env.bootstrap_train(cfg)
        algo=env.build_train(cfg)
        algo.total_steps=proto['common_schedule_horizon']
        with contextlib.redirect_stdout(io.StringIO()):algo.start_task(task,range(bpe))
        assert not cfg.ifa_enabled and cfg.train.loss_scale==1.
        assert len(algo.optimizer.state)==0 and algo._optimizer_step_count==0
        params=[n for n,p in algo.policy.named_parameters() if p.requires_grad]
        assert all(n.startswith(('module.temporal_transformer.','module.policy_head.')) for n in params)
        freeze(report/'INITIALIZATION.json',dict(base_checkpoint=cfg.pretrain,
               base_sha256=read(OUT/'EXPERIMENT_CONTEXT.json')['base_sha256'],
               model_sha256=g.digest(algo.policy.state_dict()),parameters=params,
               trainable_numel=sum(p.numel() for p in algo.policy.parameters() if p.requires_grad),
               optimizer_initial_sha256=g.digest(algo.optimizer.state_dict()),
               scheduler_initial_sha256=g.digest(algo.scheduler.state_dict()),
               frozen_reference_prefix_step=300,formal_training_resumed=False))
        recovery=STORE/'active_source_recovery.pt';position=0
        if recovery.exists():
            saved=torch.load(recovery,map_location='cpu')
            if saved['seed']==seed and saved['task']==task:
                g.restore(algo,saved['state'])
                assert all(g.compare_states(saved['state'],g.capture(algo)).values())
                position=saved['position']
            del saved
        def endpoint(step):
            model={n[7:] if n.startswith('module.') else n:v for n,v in g.cpu_state(algo.policy.state_dict()).items()}
            path=run/('step%04d.pt'%step)
            if step==0:
                shared=STORE/'bases'/('seed%d.pt'%seed)
                if shared.exists():
                    saved_base=torch.load(shared,map_location='cpu')
                    assert g.digest(saved_base['model'])==g.digest(model)
                else:env.atomic_save(shared,dict(model=model,seed=seed,step=0))
                if not path.exists():os.link(shared,path)
            else:
                env.atomic_save(path,dict(model=model,seed=seed,source=task,step=step,
                                base_sha256=read(OUT/'EXPERIMENT_CONTEXT.json')['base_sha256']))
        if position==0:
            # Duplicate first native step under identical state and random stream;
            # both paths are discarded and the real source begins from the same state.
            algo.policy.train();base=g.capture(algo)
            witnesses=[]
            for _ in range(2):
                g.restore(algo,base)
                env.seed_all(stable_seed('task_relation_forecast_v1','train',seed,task,0))
                g.native(algo,g.batch(ds,stream[0]['indices']),task,0)
                witnesses.append(g.capture(algo))
            parity=g.compare_states(*witnesses)
            assert all(parity.values()),parity
            g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values())
            write(report/'NATIVE_FIRST_STEP_PARITY.json',dict(status='PASS',checks=parity,
                  first_step_lr=proto['step_lr'][0],discarded_parity_steps=2,formal_training_resumed=False))
            endpoint(0);del base,witnesses
        begin=time.monotonic()
        for k in range(position,1000):
            if shutil.disk_usage(STORE).free < 2*1024**3:
                raise RuntimeError('STORAGE_RESERVE_REACHED')
            algo.policy.train()
            env.seed_all(stable_seed('task_relation_forecast_v1','train',seed,task,k))
            lr=algo.optimizer.param_groups[0]['lr']
            assert lr==proto['step_lr'][k],(k,lr,proto['step_lr'][k])
            value=g.native(algo,g.batch(ds,stream[k]['indices']),task,stream[k]['epoch'])
            assert algo._optimizer_step_count==k+1 and algo._scheduler_step_count==k+1
            loss=float(value[0] if isinstance(value,(tuple,list)) else value)
            assert math.isfinite(loss)
            write(report/'steps'/('%04d.json'%(k+1)),dict(step=k+1,record_indices=stream[k]['indices'],
                  lr=lr,lr_after=algo.optimizer.param_groups[0]['lr'],loss=loss,
                  optimizer_step=k+1,scheduler_step=k+1))
            if k+1 in (100,300,1000):endpoint(k+1)
            if (k+1)%100==0:
                env.atomic_save(recovery,dict(seed=seed,task=task,position=k+1,state=g.capture(algo)))
            if (k+1)%20==0:
                write(OUT/'STATUS.json',dict(protocol_status='RUNNING',stage='SOURCE_PILOT',seed=seed,task=task,
                      source_step=k+1,source_budget=1000,seconds=time.monotonic()-begin,
                      diagnostic_training_performed=True,formal_training_resumed=False))
                print('SOURCE',seed,task,k+1,loss,round(time.monotonic()-begin,2),flush=True)
        write(report/'COMPLETE.json',dict(status='PASS',seed=seed,task=task,steps=1000,
              seconds=time.monotonic()-begin,reference300=str(run/'step0300.pt'),
              diagnostic_training_performed=True,formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--task',type=int,required=True)
    a=p.parse_args();main(a.seed,a.task)
