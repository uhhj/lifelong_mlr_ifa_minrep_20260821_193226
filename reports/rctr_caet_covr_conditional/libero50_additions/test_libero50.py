import unittest
from fractions import Fraction as F
import torch
from l50_exp.metrics import metrics
from l50_exp.grouped_ifa import loss

class Properties(unittest.TestCase):
    def test_perfect_matrix(self):
        m=metrics({(s,t):F(1) for s in range(1,6) for t in range(5*s)})
        self.assertEqual(m["AUC"]["fraction"],"1/1");self.assertEqual(m["NBT"]["fraction"],"0/1")
    def test_immediate_forgetting(self):
        r={(s,t):F(int(t//5+1==s)) for s in range(1,6) for t in range(5*s)}
        m=metrics(r);self.assertEqual(m["NBT"]["fraction"],"1/1")
        self.assertEqual(m["AUC"]["fraction"],str(sum((F(1,k) for k in range(1,6)),F(0))/5))
    def test_task_not_group_omitted(self):
        r={(s,t):F(0) for s in range(1,6) for t in range(5*s)};del r[5,24]
        with self.assertRaises(AssertionError):metrics(r)
    def test_old_rows_no_ifa_gradient(self):
        z=torch.randn(7,5,requires_grad=True);ids=torch.tensor([0,0,1,1,2,2,3])
        refs={i:torch.randn(5) for i in range(4)}
        v=loss(z,ids,dict(current_tasks=[2,3],selected_pairs=[(0,2),(1,3)]),refs)
        v.backward();self.assertTrue(torch.equal(z.grad[:4],torch.zeros_like(z.grad[:4])))
    def test_single_current_scalar_formula(self):
        torch.manual_seed(9);z=torch.randn(6,4,requires_grad=True);refs={i:torch.randn(4) for i in range(3)}
        actual=loss(z,torch.tensor([2]*6),dict(current_tasks=[2],selected_pairs=[(0,2),(1,2)]),refs)
        norm=lambda x:x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        own=norm(refs[2]);g=norm(z)
        angle=lambda x:torch.acos(x.clamp(-1+1e-6,1-1e-6))
        expected=torch.stack([torch.relu(angle((g*own).sum(-1))-angle((g*norm(refs[t])).sum(-1))+
                     .1*angle((own*norm(refs[t])).sum())) for t in (0,1)]).mean()
        self.assertTrue(torch.allclose(actual,expected,rtol=0,atol=1e-7))
    def test_unpaired_current_zero(self):
        z=torch.randn(3,5,requires_grad=True)
        v=loss(z,torch.tensor([1,1,1]),dict(current_tasks=[1],selected_pairs=[]),{1:torch.randn(5)})
        v.backward();self.assertEqual(float(v),0.);self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))
if __name__=="__main__":unittest.main()
