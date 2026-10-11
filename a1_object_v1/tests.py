import unittest
import torch
from .geometry import append_basis,validate,project,angle_hinge


class GeometryTests(unittest.TestCase):
    def setUp(self):
        g=torch.Generator().manual_seed(41011)
        self.refs=[torch.randn(512,generator=g) for _ in range(8)]
        U=torch.empty(512,0,dtype=torch.float64)
        for r in self.refs:U,_=append_basis(U,r)
        self.U=U;self.z=torch.randn(5,512,generator=g)
    def loss(self,z,U):
        return angle_hinge(project(z,U),self.refs[0],self.refs[1],torch.tensor(1.5))[0].mean()
    def test_projection_and_reference_tolerances(self):
        validate(self.U,self.refs)
        z=project(self.z,self.U.float())
        self.assertLess(float((project(z,self.U.float())-z).norm()/z.norm()),1e-5)
    def test_append_preserves_old_vectors_and_duplicate(self):
        u,r=append_basis(self.U,self.refs[0])
        self.assertLess(r,1e-7);self.assertTrue(torch.equal(u,self.U))
        u,r=append_basis(self.U,torch.ones(512))
        self.assertTrue(torch.equal(u[:,:8],self.U))
    def test_identity_loss_and_gradient(self):
        z=self.z.clone().requires_grad_();I=torch.eye(512)
        a=self.loss(z,I);ga=torch.autograd.grad(a,z)[0]
        b=angle_hinge(z,self.refs[0],self.refs[1],torch.tensor(1.5))[0].mean()
        gb=torch.autograd.grad(b,z)[0]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
    def test_basis_rotation_invariance(self):
        Q=torch.linalg.qr(torch.arange(64,dtype=torch.float64).reshape(8,8)+torch.eye(8)).Q
        a=project(self.z,self.U.float());b=project(self.z,(self.U@Q).float())
        self.assertLess(float((a-b).norm()/a.norm()),1e-5)
        self.assertLess(float(abs(self.loss(self.z,self.U.float())-self.loss(self.z,(self.U@Q).float()))),1e-5)
    def test_complement_invariance_and_gradient(self):
        U=self.U.float();v=self.z-project(self.z,U)
        z=self.z.clone().requires_grad_()
        a=self.loss(z,U);b=self.loss(z+v,U)
        self.assertLess(float(abs(a-b)),1e-5)
        grad=torch.autograd.grad(a,z)[0]
        self.assertLess(float((grad-project(grad,U)).norm()/grad.norm()),1e-5)
        self.assertGreater(float(grad.norm()),0)


if __name__=='__main__':unittest.main(verbosity=2)
