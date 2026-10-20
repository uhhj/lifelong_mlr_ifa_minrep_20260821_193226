"""Synthetic numerical/gradient checks, not real task performance."""
import os
import unittest
import torch
from torch.distributions import Independent, Normal, kl_divergence
from execution_core import calibrate_distribution

DEVICE = os.environ.get('TEST_DEVICE', 'cpu')
if DEVICE == 'cuda':
    assert torch.cuda.is_available()

class ExecutionTests(unittest.TestCase):
    def data(self, dtype=torch.float32, n=128):
        g=torch.Generator(device=DEVICE).manual_seed(9281)
        b=torch.randn(n,5,7,device=DEVICE,dtype=dtype,generator=g)*2
        s=torch.logspace(-4,0,7,device=DEVICE,dtype=dtype)[None,None].expand_as(b).clone()
        l=torch.randn(n,5,device=DEVICE,dtype=dtype,generator=g)
        d=torch.randn(n,7,device=DEVICE,dtype=dtype,generator=g).tanh()*.1
        return b,s,l,d
    def test_zero_exact(self):
        for dt in (torch.float32,torch.float64):
            b,s,l,d=self.data(dt);o=calibrate_distribution(b,s,l,d*0)
            self.assertTrue(torch.equal(o['means'],b.tanh()))
            self.assertTrue(bool((o['actual_kl']==0).all()))
    def test_bound_actual_float32_and_float64(self):
        for dt in (torch.float32,torch.float64):
            b,s,l,d=self.data(dt,1024);o=calibrate_distribution(b,s,l,d)
            self.assertTrue(bool((o['actual_kl']<=.5+1e-12).all()))
            self.assertTrue(bool((o['raw_kl']>.5).any()))
    def test_unselected_and_parent_distribution_fixed(self):
        b,s,l,d=self.data();o=calibrate_distribution(b,s,l,d)
        mask=torch.ones_like(l,dtype=torch.bool);mask.scatter_(1,l.argmax(-1)[:,None],False)
        self.assertTrue(torch.equal(o['means'][mask],b.tanh()[mask]))
        self.assertTrue(torch.equal(o['scales'],s) and torch.equal(o['logits'],l))
    def test_matches_closed_form_kl(self):
        b,s,l,d=self.data();o=calibrate_distribution(b,s,l,d)
        a=Independent(Normal(o['executed_mean'].double(),o['executed_scale'].double()),1)
        p=Independent(Normal(o['base_selected'].double(),o['executed_scale'].double()),1)
        self.assertTrue(torch.allclose(kl_divergence(a,p),o['actual_kl'],atol=1e-12,rtol=1e-10))
    def test_backward_zero_and_projected(self):
        for zero in (True,False):
            b,s,l,d=self.data();d=(d*0 if zero else d).requires_grad_()
            o=calibrate_distribution(b,s,l,d)
            loss=-o['executed_dist'].log_prob(torch.zeros_like(d)).mean();loss.backward()
            self.assertTrue(torch.isfinite(d.grad).all() and bool((d.grad!=0).any()))
    def test_no_projection_ablation(self):
        b,s,l,d=self.data();o=calibrate_distribution(b,s,l,d,None)
        ix=torch.arange(len(b),device=DEVICE)
        self.assertTrue(torch.equal(o['executed_mean'],(b[ix,l.argmax(-1)]+d).tanh()))
        self.assertTrue(bool((o['alpha']==1).all()))
    def test_selected_nll_identity(self):
        b,s,l,d=self.data();o=calibrate_distribution(b,s,l,d)
        a=torch.zeros_like(d);manual=-Normal(o['executed_mean'],o['executed_scale']).log_prob(a).sum(-1)
        self.assertTrue(torch.equal(-o['executed_dist'].log_prob(a),manual))
    def test_no_rng_consumption(self):
        b,s,l,d=self.data();state=torch.cuda.get_rng_state() if DEVICE=='cuda' else torch.get_rng_state()
        calibrate_distribution(b,s,l,d)
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state() if DEVICE=='cuda' else torch.get_rng_state()))
    def test_invalid_scale_rejected(self):
        b,s,l,d=self.data();s[0,0,0]=0
        with self.assertRaises(AssertionError):calibrate_distribution(b,s,l,d)

if __name__=='__main__':
    unittest.main()
