"""Exercise the actual observe method with synthetic latent inputs and no updates."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch

tree = ast.parse((Path(__file__).resolve().parents[1] / 'libero/lifelong/algos/base.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sequential')
observe = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'observe')
norm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'l2norm')
namespace = dict(torch=torch, itertools=itertools, nn=torch.nn, DDP=type('UnusedDDP', (), {}))
exec(compile(ast.Module(body=[norm, observe], type_ignores=[]), '<actual observe>', 'exec'), namespace)

def run(fix, tasks, current=7, enabled=True):
    g = torch.tensor([[.6, .8], [.8, .6]], requires_grad=True)
    refs = torch.tensor([[1., 0.] if t == 6 else [0., 1.] for t in tasks])
    f = refs[:, None, None, None, :].expand(-1, 1, 8, 4, -1)
    cfg = NS(train=NS(grad_clip=None), ifa_enabled=enabled, feature_dir='unused', gamma=.3, ifa_weight=.1)
    if fix is not None:
        cfg.ifa_own_reference_fix = fix
    self = NS(cfg=cfg, loss_scale=1., map_tensor_to_device=lambda x:x,
              optimizer=NS(zero_grad=lambda:None), optimizer_step_and_maybe_schedule=lambda:None,
              policy=NS(compute_loss=lambda *args:(g.sum()*0, g)))
    info = dict(task_ids=[6,7], selected_pair_mask=torch.tensor([[False,True],[True,False]]),
                sim_language=torch.eye(2), sim_agent=torch.eye(2))
    with patch('torch.load', return_value={'feature':torch.tensor([1.,0.]).reshape(1,1,1,2)}):
        loss = namespace['observe'](self, [{'feature':f},tasks], current, 0, info)
    return loss, g.grad

def test_default_preserves_release():
    a,b = run(None,[6,7]),run(False,[6,7])
    assert a[0] == b[0] and torch.equal(a[1],b[1])

def test_negative_first_has_cancelled_release_but_nonzero_corrected_gradient():
    a,b = run(False,[6,7]),run(True,[6,7])
    assert a[0]>0 and a[1].norm()==0
    assert b[1].norm()>0 and b[1][0].norm()==0

def test_correct_first_is_unchanged():
    a,b = run(False,[7,6]),run(True,[7,6])
    assert a[0] == b[0] and torch.equal(a[1],b[1])

def test_task6_and_disabled_ifa_are_unchanged():
    for tasks,current,enabled in (([6,6],6,True),([6,7],7,False)):
        a,b = run(False,tasks,current,enabled),run(True,tasks,current,enabled)
        assert a[0] == b[0] and torch.equal(a[1],b[1])
