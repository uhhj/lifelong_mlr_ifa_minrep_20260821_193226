import unittest
import torch
from fractions import Fraction as F
from .model import angular_mean
from .summarize import metrics
class ProtocolTests(unittest.TestCase):
    def test_ifa_own_reference(self):
        g=torch.tensor([[1.,0.]],requires_grad=True);own=torch.tensor([1.,0.]);old=torch.tensor([0.,1.])
        self.assertEqual(angular_mean(g,own,[old],.3).item(),0.)
        self.assertGreater(angular_mean(g,old,[own],.3).item(),1.)
    def test_mean_pair_not_sum(self):
        g=torch.tensor([[0.,1.]],requires_grad=True);own=torch.tensor([1.,0.]);old=torch.tensor([0.,1.])
        self.assertTrue(torch.equal(angular_mean(g,own,[old],.3),angular_mean(g,own,[old,old],.3)))
    def test_empty_differentiable(self):
        g=torch.ones(2,2,requires_grad=True);z=angular_mean(g,g[0],[],.3);z.backward();self.assertEqual(g.grad.abs().sum().item(),0)
    def test_metrics_exact_perfect(self):
        r={(i,j):F(1) for i in range(6,10) for j in range(6,i+1)};m=metrics(r)
        self.assertEqual(m['AUC']['fraction'],'1');self.assertEqual(m['NBT']['fraction'],'0')
    def test_metrics_task_balanced(self):
        r={(i,j):F(int(j==9)) for i in range(6,10) for j in range(6,i+1)};m=metrics(r)
        self.assertEqual(m['AUC']['fraction'],'1/4');self.assertEqual(m['FWT']['fraction'],'1/4')
    def test_forgetting_sign(self):
        r={(i,j):F(int(i==j)) for i in range(6,10) for j in range(6,i+1)}
        self.assertEqual(metrics(r)['NBT']['fraction'],'1')
if __name__=='__main__':unittest.main()
