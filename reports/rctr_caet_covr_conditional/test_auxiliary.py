"""Numerical/decision software tests only; never substitute for real CUDA admission."""
import math
import unittest
import torch
from caet_exp.transfer import pair_masks, transfer_loss
from caet_exp.model import EffectModel
from covr_exp.loss import objective
from covr_exp.audit_collection import counts, qualifies


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.means = torch.nn.Parameter(torch.tensor(
            [[[0., 0.], [1., -1.]]], dtype=torch.float64))
        self.logits = torch.nn.Parameter(torch.tensor([[.2, -.3]], dtype=torch.float64))

    def forward_fn(self, z):
        n = len(z)
        return self.means.expand(n,-1,-1), torch.ones(n,2,2,dtype=z.dtype), self.logits.expand(n,-1)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = Head()

    def temporal_encode(self, x):
        return x


def outcomes(bits):
    return [(dict(success=b), None) for b in bits]


class Properties(unittest.TestCase):
    def test_pairs_exclude_self_same_task_static(self):
        e = torch.tensor([[0.,0.],[.1,0.],[.2,0.],[2.,0.]])
        p, n = pair_masks(e,torch.tensor([True,True,False,True]),
                          torch.tensor([6,7,8,6]))
        self.assertTrue(p[0,1])
        self.assertFalse(p.diagonal().any())
        self.assertFalse(p[0,2])
        self.assertFalse(p[0,3])
        self.assertTrue(n[1,3])

    def test_uncertain_pair_ignored(self):
        p,n=pair_masks(torch.tensor([[0.],[.75]]),torch.ones(2,dtype=torch.bool),
                       torch.tensor([6,7]))
        self.assertFalse(p.any() or n.any())

    def test_no_negative_no_fake_contrast(self):
        h=torch.randn(3,128,requires_grad=True)
        loss,count=transfer_loss(h,torch.zeros(3,6),torch.ones(3,dtype=torch.bool),
                                 torch.tensor([6,7,8]))
        self.assertEqual(count,0)
        self.assertEqual(float(loss),0)
        loss.backward()
        self.assertTrue(torch.equal(h.grad,torch.zeros_like(h)))

    def test_contrast_gradient_and_permutation(self):
        h=torch.nn.functional.normalize(torch.randn(4,128),dim=-1).requires_grad_()
        e=torch.tensor([[0.],[.1],[2.],[2.1]])
        m=torch.ones(4,dtype=torch.bool);t=torch.tensor([6,7,6,7])
        loss,count=transfer_loss(h,e,m,t)
        p=torch.tensor([2,0,3,1])
        other,n=transfer_loss(h[p],e[p],m[p],t[p])
        self.assertEqual(count,4);self.assertEqual(n,count)
        self.assertTrue(torch.allclose(loss,other))
        loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all() and h.grad.abs().sum()>0)

    def test_frozen_effect_keeps_action_gradient(self):
        f=EffectModel().requires_grad_(False)
        x=torch.randn(4,8,4,512)
        a=torch.randn(4,7,requires_grad=True)
        y,h=f(x,a)
        self.assertEqual(y.shape,(4,6));self.assertEqual(h.shape,(4,128))
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all() and a.grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None for p in f.parameters()))

    def test_effect_language_is_not_an_input(self):
        f=EffectModel().eval()
        x=torch.randn(2,8,4,512);a=torch.randn(2,7)
        y=f(x,a)[0];x[:,:,0]+=1000
        self.assertTrue(torch.equal(y,f(x,a)[0]))

    def test_covr_likelihood_matches_scalar_oracle(self):
        p=Policy()
        records=[dict(feature=torch.zeros(1,2,dtype=torch.float64),
                      action=torch.tensor([.4,-.2],dtype=torch.float64),
                      weight=w) for w in (1.,.25,.25,.25)]
        actual,_=objective(p,records,torch.device("cpu"))
        logits=[.2,-.3]
        densities=[math.exp(-((.4-u)**2+(-.2-v)**2)/2)/(2*math.pi)
                   for u,v in ((0,0),(1,-1))]
        def mixture(temp):
            w=[math.exp(x/temp) for x in logits]
            return -math.log(sum(a*b for a,b in zip(w,densities))/sum(w))
        expected=.1*(1.75/4)*(mixture(1)+.5*mixture(.25))
        self.assertAlmostEqual(float(actual),expected,places=10)
        actual.backward()
        self.assertTrue(p.policy_head.means.grad.abs().sum()>0)
        self.assertTrue(p.policy_head.logits.grad.abs().sum()>0)

    def test_covr_window_weight_not_renormalized(self):
        p=Policy()
        r=dict(feature=torch.zeros(1,2,dtype=torch.float64),
               action=torch.zeros(2,dtype=torch.float64),weight=1.)
        full=objective(p,[r]*4,torch.device("cpu"))[0]
        quarter=objective(p,[dict(r,weight=.25)]*4,torch.device("cpu"))[0]
        self.assertTrue(torch.allclose(quarter,full*.25))

    def test_two_rescues_required(self):
        self.assertFalse(qualifies(outcomes([0,0,0]),outcomes([1,0,0])))
        self.assertTrue(qualifies(outcomes([0,0,0]),outcomes([1,1,0])))

    def test_any_new_failure_disqualifies(self):
        self.assertEqual(counts(outcomes([0,0,1]),outcomes([1,1,0])),(2,1))
        self.assertFalse(qualifies(outcomes([0,0,1]),outcomes([1,1,0])))

    def test_no_gain_is_not_certificate(self):
        self.assertFalse(qualifies(outcomes([1,1,1]),outcomes([1,1,1])))


if __name__=="__main__":
    unittest.main(verbosity=2)
