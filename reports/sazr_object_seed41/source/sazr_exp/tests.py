import unittest,os
import torch
from .core import *

class Tests(unittest.TestCase):
    def setUp(self):
        self.device=os.environ.get("TEST_DEVICE","cpu")
        gen=torch.Generator().manual_seed(12)
        self.a=torch.randn(32,512,generator=gen,dtype=torch.float64).to(self.device)
        self.e=torch.eye(512,dtype=torch.float64,device=self.device)[:7]
        self.q,_=basis(self.a);self.b=Residual(self.q,self.e,self.e[6]).to(self.device)
    def test_anchor_zero_nonzero_uv(self):
        self.b.U.data.fill_(.7)
        self.assertLess(float(self.b(self.a.float(),self.e[6:7].repeat(32,1)).abs().max()),1e-6)
    def test_complement_nonzero(self):
        self.b.U.data.fill_(.7)
        self.assertGreater(float(self.b(torch.ones(2,512,device=self.device),self.e[6:7].repeat(2,1)).abs().max()),1e-5)
    def test_routes(self):
        self.assertEqual(self.b.route(self.e).tolist(),[False]*6+[True])
        self.assertFalse(bool(self.b.route(torch.zeros(1,512,device=self.device)).any()))
        self.assertFalse(bool(self.b.route(torch.ones(1,512,device=self.device)).any()))
    def test_zero_restores(self):
        b=torch.randn(3,5,7,device=self.device);s=torch.ones_like(b);l=torch.randn(3,5,device=self.device)
        o=corrected(b,s,l,torch.zeros(3,7,device=self.device))
        self.assertTrue(torch.equal(o["means"],b.tanh()))
    def test_other_components(self):
        b=torch.zeros(3,5,7,device=self.device);s=torch.ones_like(b);l=torch.zeros(3,5,device=self.device)
        o=corrected(b,s,l,torch.full((3,7),.03,device=self.device))
        self.assertTrue(torch.equal(o["means"][:,1:],b[:,1:]))
        self.assertGreater(float(o["means"][:,0].abs().max()),0)
    def test_noise_indexed(self):
        before=torch.get_rng_state()
        a=noise_table([30,31]);b=noise_table([31,30])
        self.assertTrue(torch.equal(a[0],b[1]));self.assertTrue(torch.equal(a[1],b[0]))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_sampling_times(self):
        ids=choose_anchor_times([100,200,300])
        self.assertEqual(list(map(len,ids)),[11,11,10])
        self.assertEqual(ids[0][0],0);self.assertEqual(ids[0][-1],99)
        self.assertEqual(sum(map(len,choose_anchor_times([1,1]))),2)
    def test_rank(self):
        q,d=basis(self.a[:1].repeat(10,1));self.assertEqual(d["rank"],1)
        self.assertEqual(q.shape,(512,1))
if __name__=="__main__":unittest.main()
