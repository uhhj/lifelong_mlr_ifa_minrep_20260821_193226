"""Role-restricted data and unchanged native model/update adapters."""
import contextlib
import csv
import io
import math
import os
import numpy as np
from .common import ROOT, OUT, STORE, read, write, sha


def allowed(purpose, role, task, source=None, target=None):
    if purpose == 'signal':
        return role == 'D_signal' and task in (source, target)
    if purpose == 'shadow':
        return role == 'A_train' and task == target
    if purpose == 'train':
        return role == 'A_train' and task == target
    if purpose == 'evaluate':
        return role == 'E_eval'
    return False


def load_data(seed, task, role, purpose, source=None, target=None):
    assert allowed(purpose, role, task, source, target), 'DATA_ROLE_ACCESS_DENIED'
    import torch
    directory = STORE/'features'/('seed%d'%seed)/('task%d'%task)/role
    ff, aa, rows, encoder = [], [], [], None
    for path in sorted(directory.glob('chunk*.npz')):
        meta = read(path.with_suffix('.json'))
        assert meta['task']==task and meta['role']==role and meta['seed']==seed
        assert sha(path)==meta['file_sha256']
        if encoder is None:encoder=meta['encoder_sha256']
        assert encoder==meta['encoder_sha256']
        with np.load(path) as z:
            ff.append(z['feature']);aa.append(z['actions'])
        rows.extend(meta['rows'])
    expected=read(OUT/'DATA_MANIFEST.json')['tasks'][str(task)]['windows'][role]
    assert rows==expected and rows, ('Incomplete or reordered data',seed,task,role)
    return dict(feature=torch.from_numpy(np.concatenate(ff)),actions=torch.from_numpy(np.concatenate(aa)),
                rows=rows,task=task,encoder_sha256=encoder)


def batch(data, indices):
    return [dict(feature=data['feature'][indices,None],actions=data['actions'][indices,None]),
            [str(data['task'])]*len(indices)]


def table(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def build(seed, task, destination):
    from cb_ifa_experiment import gpu_runtime as env
    import torch
    destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env.init_ddp(destination/('ddp_%d'%os.getpid()))
    cfg=env.train_config(seed,destination);env.bootstrap_train(cfg);algo=env.build_train(cfg)
    proto=read(OUT/'TRAINING_PROTOCOL.json')
    n=len(read(OUT/'DATA_MANIFEST.json')['tasks'][str(task)]['windows']['A_train'])
    algo.total_steps=proto['common_schedule_horizon']
    with contextlib.redirect_stdout(io.StringIO()):algo.start_task(task,range(math.ceil(n/proto['batch_size'])))
    assert not cfg.ifa_enabled and cfg.train.loss_scale==1.
    ref=read(OUT/'sources'/('s%d_t%d'%(seed,task))/'INITIALIZATION.json')
    assert env.digest(algo.optimizer.state_dict())==ref['optimizer_initial_sha256']
    assert env.digest(algo.scheduler.state_dict())==ref['scheduler_initial_sha256']
    return algo


def model_path(seed, kind, source=None, target=None, step=None):
    if kind=='base':return STORE/'bases'/('seed%d.pt'%seed)
    if kind in ('source','reference'):
        return STORE/'sources'/('s%d_t%d'%(seed,source))/('step%04d.pt'%step)
    if kind=='transfer':return STORE/'transfers'/('s%d_%d_%d'%(seed,source,target))/('step%04d.pt'%step)
    raise ValueError(kind)


def load_model(algo, path):
    import torch
    from gradient_relation_experiment import kernel as g
    state=torch.load(path,map_location='cpu')['model']
    g.module(algo).load_state_dict(state,strict=True)


@contextlib.contextmanager
def evaluation(algo):
    from cb_ifa_experiment import gpu_runtime as env
    modes={n:m.training for n,m in algo.policy.named_modules()};rng=env.rng_state()
    algo.policy.eval()
    try:yield
    finally:
        for n,m in algo.policy.named_modules():m.training=modes[n]
        env.restore_rng(rng)


def terminal(algo, data):
    import torch
    from gradient_relation_experiment import kernel as g
    with evaluation(algo),torch.no_grad():
        z=torch.cat([g.module(algo).temporal_encode(x.cuda())[:,-1,:] for x in data['feature'].split(16)])
    return z


def measure(algo, data):
    import torch
    from gradient_relation_experiment import kernel as g
    z=terminal(algo,data);head=g.module(algo).policy_head
    with evaluation(algo),torch.no_grad():
        actions=data['actions'].cuda();means,stds,logits=head.forward_fn(z)
        weights=logits.softmax(-1);expected=(weights[:,:,None]*means).sum(1)
        nll=head.loss_fn(head(z),actions,reduction='none')
        assert torch.isfinite(nll).all()
        rows=[dict(trajectory_id=r['trajectory_id'],window_index=r['window_index'],nll=float(nll[i])) for i,r in enumerate(data['rows'])]
        grouped={}
        for r in rows:grouped.setdefault(r['trajectory_id'],[]).append(r['nll'])
        return dict(mean_nll=float(np.mean([np.mean(v) for v in grouped.values()])),rows=rows,
                    expected_action_mse=float((expected-actions[:,-1,:]).square().mean()),
                    mean_mixture_weights=weights.mean(0).cpu().tolist(),
                    mean_component_scales=stds.mean(0).cpu().tolist())
