import copy
import io
import unittest
import torch
from .policy_distill import mixture,sample_teacher,loss_from_samples,deterministic_distribution


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.f=dict(log_weights=torch.log_softmax(torch.tensor([[1.,2.,3.,4.,5.]]),-1),
            means=torch.arange(35,dtype=torch.float32).reshape(1,5,7)/30,
            scales=torch.ones(1,5,7)*.2)

    def test_roundtrip_and_permutation(self):
        a,_=sample_teacher(self.f,torch.Generator().manual_seed(1))
        stream=io.BytesIO();torch.save(self.f,stream);stream.seek(0);r=torch.load(stream)
        self.assertTrue(torch.equal(mixture(self.f).log_prob(a),mixture(r).log_prob(a)))
        perm=torch.tensor([4,2,0,3,1]);q={k:v[:,perm] for k,v in self.f.items()}
        torch.testing.assert_close(mixture(self.f).log_prob(a),mixture(q).log_prob(a),rtol=1e-6,atol=3e-6)

    def test_private_rng_resume(self):
        generator=torch.Generator().manual_seed(2);normal=torch.get_rng_state();s=generator.get_state()
        a,p=sample_teacher(self.f,generator);generator.set_state(s);b,q=sample_teacher(self.f,generator)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(p,q));self.assertTrue(torch.equal(normal,torch.get_rng_state()))
        self.assertFalse(a.requires_grad);self.assertFalse(p.requires_grad)

    def test_equal_density_not_zero_mc_gradient(self):
        a,p=sample_teacher(self.f,torch.Generator().manual_seed(3))
        sf={k:v.clone().requires_grad_() for k,v in self.f.items()}
        loss,per=loss_from_samples(mixture(sf),a,p)
        self.assertEqual(float(loss),0.)
        loss.backward();self.assertTrue(torch.isfinite(sf['means'].grad).all())

    def test_no_negative_clamp(self):
        a,p=sample_teacher(self.f,torch.Generator().manual_seed(3));student=mixture(self.f)
        loss,_=loss_from_samples(student,a,p-1.)
        self.assertAlmostEqual(float(loss),-1.,places=6)

    def test_gaussian_analytic(self):
        f=dict(log_weights=torch.zeros(1,1),means=torch.zeros(1,1,7),scales=torch.ones(1,1,7))
        s=dict(log_weights=torch.zeros(1,1),means=torch.ones(1,1,7)*.3,scales=torch.ones(1,1,7)*1.2)
        a,p=sample_teacher(f,torch.Generator().manual_seed(4),100000)
        loss,_=loss_from_samples(mixture(s),a,p)
        analytic=7*(torch.log(torch.tensor(1.2))+(1+.3**2)/(2*1.2**2)-.5)
        self.assertLess(abs(float(loss-analytic)),.02)

    def test_graph_safe_mode_isolation(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.policy_head=torch.nn.Linear(2,2);self.drop=torch.nn.Dropout(.5);self.register_buffer('scale',torch.ones(2))
            def forward(self,x):return self.policy_head(self.drop(x)*self.scale)
        m=Toy();x=torch.ones(3,2);base=m(x).sum();rng=torch.get_rng_state();buffer=m.scale
        with deterministic_distribution(m):a=m(x).sum()
        (base+a).backward()
        self.assertIs(m.scale,buffer);self.assertTrue(m.drop.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main(verbosity=2)
