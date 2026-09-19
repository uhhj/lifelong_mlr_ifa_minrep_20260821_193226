import unittest, numpy as np
from .adapter import residual_parts, polish

class TestCalibration(unittest.TestCase):
    def test_kkt_components(self):
        A=np.eye(2); b=np.array([1.,-1.]); x=np.array([1.,0.])
        q=residual_parts(A,b,x); self.assertEqual(q['primal_positive'],0.); self.assertEqual(q['nonnegative_violation'],0.)
        self.assertAlmostEqual(q['complementarity'],0.)
    def test_polish(self):
        A=np.array([[2.,.2],[.2,1.]]); b=np.array([1.,.5]); x=np.zeros(2)
        q=polish(A,b,x,1000); self.assertLess(q['kkt'],1e-8)
    def test_tolerances(self):
        A=np.eye(1); self.assertTrue(residual_parts(A,np.array([1.]),np.array([1.-5e-9]))['kkt']<=1e-8)

if __name__=='__main__': unittest.main(verbosity=2)
