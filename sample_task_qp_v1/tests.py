import unittest,json
import numpy as np
from scipy.optimize import nnls
from .solver import solve
from .common import OUT,write

def project(H,u):
    result=solve(H.T@H,H.T@u,float(np.linalg.norm(u)))
    return u-H@np.array(result['lam'])

class Tests(unittest.TestCase):
    def test_zero_and_nonconflict(self):
        np.testing.assert_array_equal(project(np.zeros((3,2)),np.ones(3)),np.ones(3))
        np.testing.assert_array_equal(project(np.eye(3),-np.ones(3)),-np.ones(3))
        np.testing.assert_array_equal(project(np.eye(3),np.zeros(3)),np.zeros(3))
    def test_one_row(self):
        np.testing.assert_allclose(project(np.array([[1.],[0.]]),np.array([2.,3.])),[0,3],atol=1e-10)
    def test_opposed_rows(self):
        H=np.array([[1.,-1.],[0.,0.]])
        np.testing.assert_allclose(project(H,np.array([2.,3.])),[0,3],atol=1e-10)
        np.testing.assert_array_equal(project(H.mean(axis=1,keepdims=True),np.array([2.,3.])),[2,3])
    def test_zero_update_cone(self):
        H=np.concatenate([np.eye(3),-np.eye(3)],axis=1)
        np.testing.assert_allclose(project(H,np.array([2.,-3.,4.])),0,atol=1e-9)
    def test_average_hides_conflict(self):
        H=np.array([[1.,-3.],[0.,0.]]);u=np.array([1.,2.])
        avg=project(H.mean(axis=1,keepdims=True),u);samples=project(H,u)
        self.assertGreater(float(H[:,0]@avg),0);self.assertLessEqual(float((H.T@samples).max()),1e-9)
    def test_no_sequential_order_effect(self):
        H=np.array([[1.,0.],[1.,1.]]);u=np.ones(2)
        np.testing.assert_allclose(project(H,u),project(H[:,::-1],u),atol=1e-8)
    def test_scaling_and_duplicate_rows(self):
        H=np.array([[1.,0.,1.],[1.,1.,1.],[0.,1.,0.]])
        u=np.array([1.,-1.,2.]);d=project(H,u)
        np.testing.assert_allclose(d,project(H*np.array([3.,.4,2.]),u),atol=1e-8)
    def test_random_full_NNLS_and_cone_geometry(self):
        rng=np.random.RandomState(761103)
        for n in [2,8,32,64]:
            for repeat in range(4):
                H=rng.randn(96,n);u=rng.randn(96)
                d=project(H,u);lam=nnls(H,u,maxiter=10000)[0];other=u-H@lam
                np.testing.assert_allclose(d,other,rtol=2e-6,atol=2e-7)
                self.assertLessEqual(np.max(H.T@d),1e-6)
                self.assertLessEqual(np.linalg.norm(d),np.linalg.norm(u)+1e-7)
                self.assertLess(abs(d@(u-d)),1e-6)
                # Grouping samples into four task means relaxes the feasible set.
                if n>=8:
                    means=np.stack([H[:,ix].mean(axis=1) for ix in np.array_split(np.arange(n),4)],axis=1)
                    relaxed=project(means,u)
                    self.assertLessEqual(np.linalg.norm(relaxed-u),np.linalg.norm(d-u)+1e-6)
    def test_invalid_inputs_fail(self):
        with self.assertRaises(AssertionError):solve(np.eye(2),np.array([np.nan,0]),1)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(OUT/'UNIT_TESTS.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)))
    raise SystemExit(0 if r.wasSuccessful() else 1)
