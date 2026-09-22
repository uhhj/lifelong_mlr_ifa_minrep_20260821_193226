import itertools,unittest
from unittest.mock import patch
import numpy as np
from .refine import run,RULES
from .numeric import strict,parts
class Tests(unittest.TestCase):
    def test_unchanged_pass(self):
        A=np.eye(2);b=np.array([1.,2.]);x=b.copy()
        y,m,_,_=run(A,b,x)
        self.assertTrue(np.array_equal(x,y));self.assertEqual(m['linear_solves'],0)
        self.assertEqual(m['status'],'UNCHANGED_STANDARD_PASS')
    def test_reenter_all_inactive(self):
        A=np.eye(2);b=np.array([1.,2.]);x=np.array([1.,0.])
        y,m,t,_=run(A,b,x)
        self.assertTrue(m['strict_pass']);self.assertTrue(any(r['action']=='ENTER' for r in t))
        np.testing.assert_array_equal(y,b)
    def test_remove_negative_stationary_variable(self):
        A=np.array([[2.,1.],[1.,2.]]);b=np.array([1.,-1.]);x=np.ones(2)
        y,m,t,_=run(A,b,x)
        self.assertTrue(m['strict_pass']);np.testing.assert_allclose(y,[.5,0.])
        self.assertTrue(any(r['action']=='LEAVE' for r in t))
    def test_singular_is_not_fake_pass(self):
        y,m,_,_=run(np.zeros((2,2)),np.ones(2),np.ones(2))
        self.assertFalse(m['strict_pass']);self.assertEqual(m['status'],'SINGULAR_WORKING_SET_UNRESOLVED')
    def test_iteration_budget(self):
        with patch.dict(RULES,max_working_set_iterations=0):
            _,m,_,_=run(np.eye(2),np.ones(2),np.zeros(2))
        self.assertEqual(m['status'],'ITERATION_LIMIT');self.assertFalse(m['strict_pass'])
    def test_inputs_not_modified_and_deterministic(self):
        A=np.array([[1.,.99999],[.99999,1.]]);b=np.array([.1,.2]);x=np.ones(2)
        copies=[v.copy() for v in (A,b,x)]
        y,m,t,p=run(A,b,x);z,n,u,q=run(A,b,x)
        self.assertTrue(m['strict_pass']);np.testing.assert_array_equal(y,z);np.testing.assert_array_equal(p,q)
        for old,new in zip(copies,(A,b,x)):np.testing.assert_array_equal(old,new)
    def test_cycle_detection(self):
        fake=lambda B,c:(-np.ones(len(c)),dict(condition_number=1.))
        with patch('qp_workingset_refinement_diag_v1.refine.linear',side_effect=fake):
            _,m,_,_=run(np.eye(1),np.array([2.]),np.array([1.]))
        self.assertEqual(m['status'],'CYCLE_DETECTED');self.assertFalse(m['strict_pass'])
    def test_against_exhaustive_small_active_sets(self):
        rng=np.random.RandomState(20260922)
        for case in range(12):
            B=rng.randn(4,4);A=B.T@B+np.eye(4)*.2;b=rng.randn(4)
            valid=[]
            for mask in itertools.product((0,1),repeat=4):
                W=np.flatnonzero(mask);z=np.zeros(4)
                if len(W):z[W]=np.linalg.solve(A[np.ix_(W,W)],b[W])
                if parts(A,b,z)['kkt']<=1e-8:valid.append(z)
            self.assertEqual(len(valid),1)
            y,m,_,_=run(A,b,np.abs(rng.randn(4)))
            self.assertTrue(m['strict_pass'],m);np.testing.assert_allclose(y,valid[0],rtol=1e-8,atol=1e-10)
if __name__=='__main__':unittest.main()
