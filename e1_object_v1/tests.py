import copy
import tempfile
import unittest
from pathlib import Path

import torch

from .control import Controller, Settings, put, rng_state
from .io import save


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p=torch.nn.Parameter(torch.tensor([1.,2.]))
        self.frozen=torch.nn.Parameter(torch.tensor([3.]),requires_grad=False)
        self.dropout=torch.nn.Dropout(.5)
    def compute_loss(self,data,*args,**kwargs):
        return (self.dropout(self.p)-data).square().sum().reshape(1), None


class Tests(unittest.TestCase):
    def run_case(self, settings):
        m=Toy();c=Controller(m,42,settings);count=[9]
        def native():
            with torch.no_grad(): m.p.add_(.1)
            count[0]+=1
            return 3.
        _,r=c.update(native,lambda:count[0],lambda rng,n:(torch.zeros(2),['old']),dict(task=7))
        return m,c,r,count
    def test_disabled(self):
        m,c,r,n=self.run_case(Settings(enabled=False));self.assertTrue(torch.equal(m.p,torch.tensor([1.1,2.1])));self.assertIsNone(r);self.assertEqual(n[0],10)
    def test_zero_rho_and_rng(self):
        rng=torch.get_rng_state().clone();m,c,r,n=self.run_case(Settings(rho=0.));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(m.p,torch.tensor([1.1,2.1])));self.assertFalse(r['accepted'])
    def test_accepted_and_global_cap(self):
        m,c,r,n=self.run_case(Settings());self.assertTrue(r['triggered']);self.assertTrue(r['accepted']);self.assertLessEqual(r['correction_norm'],.25*r['delta_norm']*(1+2e-6));self.assertEqual(n[0],10);self.assertEqual(m.frozen.item(),3.)
    def test_rejected_restores_candidate(self):
        m,c,r,n=self.run_case(Settings(tau_num=100.));self.assertEqual(r['reason'],'RESTORED_NATIVE_CANDIDATE');self.assertTrue(torch.equal(m.p,torch.tensor([1.1,2.1])))
    def test_repeat_and_grad_isolation(self):
        m=Toy();m.p.grad=torch.ones_like(m.p)*17;c=Controller(m,42)
        a=c.losses(torch.zeros(2));b=c.losses(torch.zeros(2));self.assertTrue(torch.equal(a,b));c.losses(torch.zeros(2),True)
        self.assertTrue(torch.equal(m.p.grad,torch.ones_like(m.p)*17));self.assertTrue(m.dropout.training)
    def test_save_load_final(self):
        m,c,r,n=self.run_case(Settings())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';save(p,m.state_dict());m2=Toy();m2.load_state_dict(torch.load(p));self.assertTrue(torch.equal(m.p,m2.p))
    def test_loss_absolute_scale(self):
        m=Toy();m.compute_loss=lambda *a,**kw:(torch.tensor([-2.,4.]),None)
        c=Controller(m,42);count=[9]
        def native():count[0]+=1;return 0.
        _,r=c.update(native,lambda:count[0],lambda *a:(torch.zeros(2),[]),{})
        self.assertAlmostEqual(r['epsilon'],3e-4)


if __name__=='__main__':unittest.main()
