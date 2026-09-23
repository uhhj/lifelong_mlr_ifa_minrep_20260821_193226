import unittest,torch
from .fisher import ModelFisher,sample
from gn_functional_geometry_v1.engine import Engine
class TestFisher(unittest.TestCase):
 def test_operator_and_rng(self):
  from cb_ifa_experiment import gpu_runtime
  from libero.lifelong.models.policy_head import GMMHead
  torch.manual_seed(23);head=GMMHead(4,2,hidden_size=8,num_layers=1,num_modes=2).double().eval()
  z=torch.randn(3,4,dtype=torch.float64);e=Engine(head,{'old_build':z},{'old_build':torch.zeros(3,2,dtype=torch.float64)},torch.ones(2,dtype=torch.float64))
  before=torch.get_rng_state().clone();draws,modes=sample(e,n=8);self.assertTrue(torch.equal(before,torch.get_rng_state()))
  f=ModelFisher(e,draws);J=torch.autograd.functional.jacobian(f.scores_output,e.theta);F=J.T@J/len(J)
  v=torch.randn_like(e.theta);v/=v.norm();self.assertTrue(torch.allclose(f.mv(v),F@v,atol=1e-10,rtol=1e-9))
  self.assertGreaterEqual(float(v@f.mv(v)),-1e-12)
  i=0;scale_energy=0.
  for name,size in zip(e.names,e.sizes):
   if 'logstd_layer' in name:scale_energy+=float(J[:,i:i+size].square().sum())
   i+=size
  self.assertGreater(scale_energy,0.)
if __name__=='__main__':unittest.main()
