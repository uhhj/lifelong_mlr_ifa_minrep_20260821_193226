import copy,unittest
from types import SimpleNamespace
import numpy as np,torch
from gradient_relation_experiment import kernel as g
from .materialize import apply,bins
class Tests(unittest.TestCase):
 def example(self):
  p=torch.nn.Parameter(torch.tensor([1.,1.],dtype=torch.float32));base=p.detach().clone();ulp=torch.nextafter(base,torch.full_like(base,float('inf')))-base
  native=base+ulp;c=ulp*torch.tensor([.25,.75]);u=native-base;d=u.double()-c.double();return p,base,native,c,d,u-c
 def test_1_materialization_flip(self):
  p,b,v,c,d,d32=self.example();h=torch.tensor([1.,-3.],dtype=torch.float64)
  self.assertEqual(float(h@d32.double()),0);w=apply([p],[b],'W0',d,d32,c,[v]);self.assertGreater(float(h@w/(h.norm()*(v-b).double().norm())),5e-6)
 def test_2_exact_reread(self):
  p,b,v,c,d,d32=self.example();w=apply([p],[b],'W1',d,d32,c,[v]);self.assertTrue(torch.equal(w,p.detach().double()-b.double()));self.assertTrue(torch.equal(p.detach(),(b.double()+d).float()))
 def test_3_state_restoration(self):
  p,b,v,c,d,d32=self.example();policy=torch.nn.Module();policy.register_parameter('p',p);opt=torch.optim.AdamW(policy.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:1.)
  p.grad=torch.ones_like(p);opt.step();scheduler.step();algo=SimpleNamespace(policy=policy,optimizer=opt,scheduler=scheduler,_optimizer_step_count=1,_scheduler_step_count=1);state=g.capture(algo)
  for method in ['W0','W1','W2']:
   apply([p],[b],method,d,d32,c,[v]);g.restore(algo,state);self.assertTrue(all(g.compare_states(state,g.capture(algo)).values()))
 def test_4_residual_identity(self):
  p,b,v,c,d,d32=self.example();h=torch.tensor([1.,-3.],dtype=torch.float64);w=apply([p],[b],'W1',d,d32,c,[v]);self.assertEqual(float(h@w),float(h@d+h@(w-d)))
 def test_5_deterministic(self):
  p,b,v,c,d,d32=self.example()
  for method in ['W1','W2']:
   with torch.no_grad():p.copy_(b)
   first=apply([p],[b],method,d,d32,c,[v])
   with torch.no_grad():p.copy_(b)
   second=apply([p],[b],method,d,d32,c,[v]);self.assertTrue(torch.equal(first,second))
 def test_bins_boundaries(self):self.assertEqual(bins(torch.tensor([0.,.5,1.,2.,4.],dtype=torch.float64)),[1,1,1,1,1])
if __name__=='__main__':unittest.main()
