"""Explicit six-task diagnostic adapters; formal runtime remains read-only."""
import contextlib
import io
import math
import os
import numpy as np
from .common import *
from task_relation_forecast_v1.runtime import batch, evaluation, load_model

REPORT=OUT/'confirmation'
DATA=STORE/'confirmation'

def entry(task):return read(OUT/'CONFIRMATION_DATA_MANIFEST.json')['tasks'][str(task)]

def configure(seed,task,destination):
    from cb_ifa_experiment import gpu_runtime as env
    # Use the historical working configuration, changing only identity and paths.
    cfg=env.train_config(45,destination)
    cfg.seed=seed;cfg.benchmark_name=entry(task)['suite']
    cfg.feature_dir=str(DATA/'features'/('seed%d'%seed));cfg.use_wandb=False
    return cfg

def load(seed,task,role,purpose):
    import torch
    allowed={'train':{'A_train'},'signal':{'D_signal'},'evaluate':{'E_eval'}}
    assert role in allowed[purpose]
    completion=read(REPORT/('CACHE_s%d_t%d_%s.json'%(seed,task,role)))
    assert completion['status']=='PASS'
    ff=[];aa=[];rows=[]
    for path in sorted((DATA/'features'/('seed%d'%seed)/('task%d'%task)/role).glob('chunk*.npz')):
        meta=read(path.with_suffix('.json'))
        assert meta['seed']==seed and meta['task']==task and meta['role']==role
        assert meta['file_sha256']==sha(path) and meta['encoder_sha256']==completion['encoder_sha256']
        with np.load(path) as z:ff.append(z['feature']);aa.append(z['actions'])
        rows.extend(meta['rows'])
    assert rows==entry(task)['windows'][role] and rows
    return dict(feature=torch.from_numpy(np.concatenate(ff)),actions=torch.from_numpy(np.concatenate(aa)),rows=rows,task=task)

def halves(data):
    split=read(REPORT/'DATA_RULES.json')['tasks'][str(data['task'])]
    return np.array([0 if r['trajectory_id'] in split['D1'] else 1 for r in data['rows']])

def build(seed,task,destination):
    import torch
    from cb_ifa_experiment import gpu_runtime as env
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    destination.mkdir(parents=True,exist_ok=True)
    env.init_ddp(destination/('ddp_%d'%os.getpid()))
    cfg=configure(seed,task,destination);env.bootstrap_train(cfg);algo=env.build_train(cfg)
    proto=read(REPORT/'TRAINING_PROTOCOL.json')
    algo.total_steps=proto['common_schedule_horizon']
    bpe=math.ceil(len(entry(task)['windows']['A_train'])/proto['batch_size'])
    with contextlib.redirect_stdout(io.StringIO()):algo.start_task(task,range(bpe))
    assert not cfg.ifa_enabled and cfg.train.loss_scale==1.
    assert not algo.optimizer.state and algo._optimizer_step_count==algo._scheduler_step_count==0
    assert all(n.startswith(('module.temporal_transformer.','module.policy_head.')) for n,p in algo.policy.named_parameters() if p.requires_grad)
    return algo

def model_path(seed,kind,source=None,target=None,step=1000):
    if kind=='base':return DATA/'bases'/('seed%d.pt'%seed)
    if kind in ('source','reference'):return DATA/'sources'/('s%d_t%d'%(seed,source))/('step%04d.pt'%step)
    if kind=='transfer':return DATA/'transfers'/unit(seed,source,target)/('step%04d.pt'%step)
    raise ValueError(kind)

def data_firewall():
    import sys
    accessed=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0]).replace('\\','/')
        if 'E_eval' in path or '/behavior/' in path or path.endswith(('TARGETS.csv','ROLLOUTS.csv')):
            raise PermissionError('CONFIRMATION_ROLE_FIREWALL: '+path)
        if 'diagnostic_storage/' in path or 'reports/' in path:accessed.add(path)
    sys.addaudithook(audit)
    return accessed
