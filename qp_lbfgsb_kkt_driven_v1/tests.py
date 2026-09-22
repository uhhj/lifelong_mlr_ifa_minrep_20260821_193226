"""Synthetic tests cover selection semantics only; not scientific fixtures."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from . import fallback as f

class SelectionTests(unittest.TestCase):
    def invoke(self,iterates,final,success=True):
        A=np.eye(1);b=np.ones(1);x=np.zeros(1)
        def fake(fun,x0,jac,bounds,method,options,callback):
            self.assertEqual(method,'L-BFGS-B');self.assertEqual(bounds,[(0.0,None)])
            self.assertEqual(options,dict(maxiter=100000,maxfun=200000,ftol=0.0,gtol=0.0,maxls=50,maxcor=1))
            for v in iterates:callback(np.array([v],dtype=float))
            return SimpleNamespace(x=np.array([final]),success=success,message='test',nit=len(iterates),nfev=len(iterates)+1,njev=len(iterates)+1)
        with patch.object(f,'minimize',side_effect=fake):r=f.run(A,b,x)
        np.testing.assert_array_equal(x,[0]);np.testing.assert_array_equal(A,np.eye(1));np.testing.assert_array_equal(b,[1])
        return r
    def test_first_pass_survives_worse_final(self):
        x,_,trace,m=self.invoke([.5,1,.8],.8)
        self.assertTrue(m['accepted']);self.assertEqual(m['first_pass_iteration'],2);self.assertEqual(x[0],1);self.assertEqual(len(trace),3)
    def test_first_pass_priority_over_later_pass(self):
        x,_,_,m=self.invoke([1-1e-9,1],1)
        self.assertEqual(m['first_pass_iteration'],1);self.assertEqual(x[0],1-1e-9)
    def test_no_first_pass_final_can_pass(self):
        _,_,_,m=self.invoke([.5,.8],1,False)
        self.assertTrue(m['accepted']);self.assertIsNone(m['first_pass_iteration']);self.assertEqual(m['selected_source'],'final_x')
    def test_success_flag_cannot_override_kkt(self):
        _,_,_,m=self.invoke([.5,.8],.8,True)
        self.assertFalse(m['accepted']);self.assertTrue(m['success'])
    def test_no_near_threshold_exception(self):
        _,_,_,m=self.invoke([1-1.009e-8],1-1.009e-8)
        self.assertFalse(m['accepted']);self.assertIsNone(m['first_pass_iteration'])
    def test_complementarity_can_dominate(self):
        p=f.parts(np.eye(1),np.ones(1),np.array([2.]));self.assertEqual(p['primal_positive'],0);self.assertEqual(p['nonnegative_violation'],0);self.assertEqual(p['complementarity'],2);self.assertEqual(p['kkt'],2)

if __name__=='__main__':unittest.main()
