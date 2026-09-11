"""Unit checks use the exact historical core, including clipping/AdamW order."""
import copy
import unittest
import torch
from libero.lifelong.algos.fixed_kappa import project_parameter_grads_,backward_mixed_with_reference,mixed_losses
from confirmation_experiment.common import exact,rng_state

class CoreTests(unittest.TestCase):
    def test_endpoints_and_soft(self):
        for k in (0.,.5,1.):
            p=torch.nn.Parameter(torch.tensor([1.,2.]));p.grad=torch.tensor([-2.,3.]);r=torch.tensor([1.,0.])
            m=project_parameter_grads_((p,),(r,),kappa=k)
            torch.testing.assert_close(p.grad,torch.tensor([-2+2*k,3.]),rtol=0,atol=0)
            self.assertEqual(m['projected'],k>0)
    def test_nonconflict_and_zero_reference(self):
        for ref in (torch.tensor([1.,0.]),torch.zeros(2)):
            p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([2.,3.]);old=p.grad.clone()
            m=project_parameter_grads_((p,),(ref,),kappa=.5)
            self.assertFalse(m['projected']);self.assertTrue(torch.equal(old,p.grad))
    def test_no_old_single_backward(self):
        p=torch.nn.Parameter(torch.tensor([2.]));m=backward_mixed_with_reference((p*p).sum(),None,(p,),kappa=.5)
        self.assertFalse(m['has_reference']);self.assertEqual(p.grad.item(),4)
    def test_mixed_loss_original_weights(self):
        x=torch.tensor([2.,4.,6.,8.],requires_grad=True)
        loss,new,old,nn,no=mixed_losses(x,[6,7,7,7],7)
        self.assertEqual(loss.item(),5);self.assertEqual(new.item(),6);self.assertEqual(old.item(),2)
        self.assertEqual((nn,no),(3,1))
    def test_clip_after_projection_and_scheduler_after_optimizer(self):
        p=torch.nn.Parameter(torch.tensor([0.,0.]));p.grad=torch.tensor([-20.,30.])
        opt=torch.optim.AdamW([p],lr=.1,weight_decay=.1)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:(n+1)/10)
        lr=opt.param_groups[0]['lr'];old=p.detach().clone()
        project_parameter_grads_((p,),(torch.tensor([1.,0.]),),kappa=.5)
        before=p.grad.clone();torch.nn.utils.clip_grad_norm_([p],1.)
        self.assertLessEqual(float(p.grad.norm()),1.000001)
        self.assertTrue(torch.allclose(p.grad,before/(before.norm()+1e-6),rtol=1e-6,atol=0))
        opt.step();self.assertEqual(float(opt.state[p]['step']),1)
        expected=old*(1-lr*.1)-lr*p.grad/(p.grad.abs()+1e-8)
        torch.testing.assert_close(p,expected,atol=1e-7,rtol=1e-6)
        sch.step();self.assertEqual(sch.last_epoch,1);self.assertNotEqual(lr,opt.param_groups[0]['lr'])
    def test_diagnostics_no_rng_draw(self):
        p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([-2.,3.]);state=rng_state()
        project_parameter_grads_((p,),(torch.tensor([1.,0.]),),kappa=.5)
        exact(state,rng_state())
if __name__=='__main__':unittest.main()
