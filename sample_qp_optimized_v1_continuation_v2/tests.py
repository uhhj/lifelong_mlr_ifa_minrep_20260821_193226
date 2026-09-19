"""CPU correctness tests. These are not a substitute for real-policy GPU parity."""
import ast
import copy
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls
from .core import Workspace, inner, project, solve


def original_functions():
    path = Path(__file__).resolve().parents[1]/'sample_task_qp_v1/measure.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ['inner', 'norm', 'sample_project']]
    # Execute the actual legacy pure math functions without importing GPU assets.
    ns = dict(torch=torch, np=np, solve=solve, nnls=nnls)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a == b


class ToyGMM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Tanh(),torch.nn.Linear(7,10))
        self.unused = torch.nn.Parameter(torch.zeros(2))

    def loss(self, data):
        x, y = data
        v = self.layers(x)
        logits, means, scales = v[:,:2], v[:,2:6].reshape(-1,2,2), v[:,6:].reshape(-1,2,2).clamp(-3,2).exp()
        d = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
            torch.distributions.Independent(torch.distributions.Normal(means, scales),1))
        return -d.log_prob(y)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.legacy = original_functions()

    def setUp(self):
        torch.manual_seed(7821)

    def test_snapshot_does_not_alias_parameters(self):
        p = torch.nn.Parameter(torch.randn(9))
        w = Workspace([p]);before = w.snapshot_parameters();saved=before[0].clone()
        with torch.no_grad():p.add_(1)
        self.assertTrue(torch.equal(before[0],saved))
        self.assertTrue(torch.equal(w.displacement(before),p.detach()-saved))

    def test_gradient_values_unused_params_and_no_rng_or_grad_mutation(self):
        model=ToyGMM();ps=tuple(model.parameters());w=Workspace(ps)
        batches=[(torch.randn(8,3),torch.randn(8,2)) for _ in range(2)]
        for p in ps:p.grad=torch.randn_like(p)
        previous=[p.grad.clone() for p in ps];rng=torch.get_rng_state().clone()
        H=w.sample_gradients(batches,model.loss,16).clone();expected=[]
        for b in batches:
            losses=model.loss(b)
            for i in range(8):
                grads=torch.autograd.grad(losses[i],ps,retain_graph=i<7,allow_unused=True)
                expected.append(torch.cat([(torch.zeros_like(p) if z is None else z).reshape(-1)
                                           for p,z in zip(ps,grads)]))
        self.assertTrue(torch.equal(H,torch.stack(expected,dim=1)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad,x) for p,x in zip(ps,previous)))

    def test_workspace_refreshes_all_columns_and_reuses_allocation(self):
        model=ToyGMM();w=Workspace(model.parameters());b=[(torch.randn(8,3),torch.randn(8,2))]
        first=w.sample_gradients(b,model.loss,8);address=first.data_ptr();old=first.clone()
        with torch.no_grad():next(model.parameters()).add_(.5)
        # The unused parameter above has no effect; change an actual layer as well.
        with torch.no_grad():model.layers[0].weight.add_(.1)
        new=w.sample_gradients(b,model.loss,8)
        self.assertEqual(address,new.data_ptr());self.assertFalse(torch.equal(old,new))

    def test_zero_nonconflict_conflict_duplicate_and_opposed_constraints(self):
        for H,u in [(torch.zeros(3,2),torch.ones(3)),(torch.eye(3),-torch.ones(3)),
                    (torch.eye(3),torch.zeros(3)),(torch.eye(3),torch.ones(3)),
                    (torch.tensor([[1.,-1.,1.],[0.,0.,0.]]),torch.tensor([2.,3.]))]:
            K,_=inner(H,gram=True)
            old,s0=self.legacy['sample_project'](H,K,u,True)
            new,s1=project(H,K,u)
            self.assertTrue(torch.equal(old,new))
            self.assertEqual(s0['lam'],s1['lam'])
            self.assertLessEqual(s1['written_normalized_rate_max'],5e-6)

    def test_chunk_boundary_and_64_columns_exactly_match_original(self):
        H=torch.randn(65539,64);u=torch.randn(65539)
        K,f=inner(H,u,gram=True);oldK,oldf=self.legacy['inner'](H,u,gram=True)
        np.testing.assert_array_equal(K,oldK);np.testing.assert_array_equal(f,oldf)
        c,_=project(H,K,u);old,_=self.legacy['sample_project'](H,K,u,False)
        self.assertTrue(torch.equal(c,old))

    def test_joint_constraints_are_not_aggregate_screened(self):
        H=torch.tensor([[1.,-3.],[0.,0.]]);u=torch.tensor([1.,2.]);K,_=inner(H,gram=True)
        self.assertLess(float(H.mean(1)@u),0)
        c,_=project(H,K,u)
        self.assertGreater(float(c.norm()),0)
        self.assertLessEqual(float((H.T@(u-c)).max()),5e-6)

    def test_parameter_correction_preserves_adam_moments_gradients_and_rng(self):
        model=ToyGMM();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.1)
        loss=model.loss((torch.randn(8,3),torch.randn(8,2))).mean();loss.backward();opt.step()
        state=copy.deepcopy(opt.state_dict())
        gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
        w=Workspace(model.parameters());base=w.snapshot_parameters();c=torch.randn(w.size)*.01
        rng=torch.get_rng_state().clone()
        w.subtract_correction(c)
        self.assertTrue(state_equal(state,opt.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for p,g0 in zip(model.parameters(),gradients):self.assertTrue(state_equal(p.grad,g0))
        expected=torch.cat([p.reshape(-1) for p in base])+(-c)
        self.assertTrue(torch.equal(torch.cat([p.detach().reshape(-1) for p in model.parameters()]),expected))

    def test_memory_and_bad_reference_fail_closed(self):
        model=ToyGMM();w=Workspace(model.parameters(),max_reference_bytes=1)
        with self.assertRaises(AssertionError):w.sample_gradients([],model.loss,8)
        w=Workspace(model.parameters());batch=[(torch.randn(7,3),torch.randn(7,2))]
        with self.assertRaises(AssertionError):w.sample_gradients(batch,model.loss,7)

    def test_twenty_adamw_updates_match_legacy_parameters_moments_scheduler(self):
        a=ToyGMM();b=copy.deepcopy(a);pa=tuple(a.parameters());pb=tuple(b.parameters())
        oa=torch.optim.AdamW(pa,lr=1e-4,weight_decay=.1);ob=torch.optim.AdamW(pb,lr=1e-4,weight_decay=.1)
        sa=torch.optim.lr_scheduler.LambdaLR(oa,lambda i:1-i/100)
        sb=torch.optim.lr_scheduler.LambdaLR(ob,lambda i:1-i/100)
        wa=Workspace(pa);wb=Workspace(pb)
        reference=[(torch.randn(8,3),torch.randn(8,2))]
        training=[(torch.randn(10,3),torch.randn(10,2)) for _ in range(20)]
        for batch in training:
            ba=wa.snapshot_parameters();bb=wb.snapshot_parameters()
            H=wa.sample_gradients(reference,a.loss,8);H2=wb.sample_gradients(reference,b.loss,8)
            self.assertTrue(torch.equal(H,H2));K,_=inner(H,gram=True)
            for model,opt,sched in [(a,oa,sa),(b,ob,sb)]:
                opt.zero_grad();model.loss(batch).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),100);opt.step();sched.step()
            d=wa.displacement(ba);c,_=self.legacy['sample_project'](H,K,d,False)
            # Legacy assignment goes through a CPU native model state.
            saved={k:v.detach().cpu().clone() for k,v in a.state_dict().items()};offset=0
            with torch.no_grad():
                for n,p in a.named_parameters():
                    p.copy_(saved[n]+(-c[offset:offset+p.numel()].reshape_as(p)));offset+=p.numel()
            d2=wb.displacement(bb);c2,_=project(H2,K,d2);wb.subtract_correction(c2)
            self.assertTrue(state_equal(a.state_dict(),b.state_dict()))
            self.assertTrue(state_equal(oa.state_dict(),ob.state_dict()))
            self.assertEqual(sa.state_dict(),sb.state_dict())


if __name__=='__main__':
    unittest.main(verbosity=2)
