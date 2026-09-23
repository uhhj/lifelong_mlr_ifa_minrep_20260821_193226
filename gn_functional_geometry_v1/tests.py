import unittest,torch,numpy as np
from .engine import Engine
class TestNumerics(unittest.TestCase):
 def test_matrix_free_explicit(self):
  from cb_ifa_experiment import gpu_runtime as env
  from libero.lifelong.models.policy_head import GMMHead
  torch.manual_seed(17)
  h=GMMHead(input_size=4,output_size=2,hidden_size=8,num_layers=1,num_modes=2).double()
  z=torch.randn(3,4,dtype=torch.float64);a=torch.zeros(3,2,dtype=torch.float64)
  e=Engine(h,{'old_build':z},{'old_build':a},torch.tensor([2.,3.],dtype=torch.float64))
  J=torch.autograd.functional.jacobian(lambda t:e.output(t,'old_build'),e.theta).reshape(6,-1)
  C=J.T@torch.diag(e.weight.repeat(3))@J/3
  v=torch.randn_like(e.theta);v=v/v.norm();expected=C@v
  self.assertTrue(torch.allclose(e.cv(v),expected,atol=1e-10,rtol=1e-9))
  self.assertAlmostEqual(e.trace(),float(C.trace()),places=10)
  original=e.theta.clone();e.output(e.theta+.001*v,'old_build')
  self.assertTrue(torch.equal(original,e.theta))
 def test_split(self):
  from .common import read,OUT
  s=read(OUT/'DATA_SPLIT.json')['roles'];sets={k:set(r['trajectory_id'] for r in v) for k,v in s.items()}
  for a,b in [('old_build','old_probe'),('old_gradient','old_probe'),('old_build','old_gradient'),('new_gradient','new_probe')]:self.assertFalse(sets[a]&sets[b])

class TestOriginalLoss(unittest.TestCase):
 def test_original_nll_and_directional_derivative(self):
  from cb_ifa_experiment import gpu_runtime as env
  from libero.lifelong.models.policy_head import GMMHead
  torch.manual_seed(31);h=GMMHead(4,2,hidden_size=8,num_layers=1,num_modes=2).double().eval()
  z=torch.randn(5,4,dtype=torch.float64);a=torch.randn(5,2,dtype=torch.float64)
  e=Engine(h,{'old_build':z},{'old_build':a},torch.ones(2,dtype=torch.float64))
  direct=h.loss_fn(h(z),a[:,None,:],reduction='none')
  self.assertTrue(torch.equal(e.losses(e.theta,'old_build'),direct))
  self.assertTrue(torch.allclose(e.output(e.theta,'old_build'),h(z).mean,atol=1e-14,rtol=1e-14))
  v=torch.randn_like(e.theta);v/=v.norm();d=float(e.grad('old_build')@v);eps=1e-6
  finite=float((e.losses(e.theta+eps*v,'old_build').mean()-e.losses(e.theta-eps*v,'old_build').mean())/(2*eps))
  self.assertAlmostEqual(d,finite,places=7)

if __name__=='__main__':unittest.main()
