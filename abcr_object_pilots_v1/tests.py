import unittest
import numpy as np
from scipy.spatial.distance import cdist
from .analyze import action_distance,paired_ci,retrieval_rows
from .common import normalize

class Contracts(unittest.TestCase):
    def test_normalization_retains_signed_cosine(self):
        a=normalize([[3.,4.],[-3.,-4.]])
        self.assertAlmostEqual(float(a[0]@a[1]),-1.)

    def test_motion_formula_all_seven_coordinates(self):
        a=np.ones((2,6,7));b=np.zeros((2,6,7));scale=np.ones(7)*2
        self.assertTrue(np.all(action_distance(a,b,np.array([0,1]),np.array([1,0]),4,scale)==.5))

    def test_single_action_agreement_can_diverge(self):
        a=np.zeros((1,6,7));b=a.copy();b[:,1:]=2
        self.assertEqual(float(action_distance(a,b,[0],[0],1,np.ones(7))[0]),0.)
        self.assertGreater(float(action_distance(a,b,[0],[0],4,np.ones(7))[0]),1.)

    def test_blocked_exact_pair_matrix(self):
        rng=np.random.RandomState(42);a=rng.randn(15,9);b=rng.randn(24,9)
        blocked=np.concatenate([cdist(a[s:s+4],b) for s in range(0,len(a),4)])
        direct=np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(2))
        self.assertTrue(np.allclose(blocked,direct,rtol=1e-14,atol=1e-14))
        self.assertAlmostEqual(float(np.quantile(blocked,.15)),float(np.quantile(direct,.15)))

    def test_paired_bootstrap_uses_input_trajectory_units(self):
        a=paired_ci([-.1,-.2,-.3],repeats=40)
        self.assertEqual(a,paired_ci([-.1,-.2,-.3],repeats=40))
        self.assertEqual(a['n'],3)
        self.assertLess(a['ci95'][1],0)

    def test_missing_neighbor_and_target_do_not_become_zero_error(self):
        d=dict(task=np.array([6,0]),demo=np.array([0,0]),t=np.array([0,0]),
               eef=np.zeros((2,3)),target_relative=np.zeros((2,3)),target_valid=np.array([True,False]))
        r=retrieval_rows(d,np.array([0]),np.array([1]),[-1],'test',6,.5)[0]
        self.assertIsNone(r['eef_error_m']);self.assertFalse(r['available'])
        r=retrieval_rows(d,np.array([0]),np.array([1]),[0],'test',6,.5)[0]
        self.assertEqual(r['eef_error_m'],0.)
        self.assertIsNone(r['object_relative_vector_error_m'])

if __name__=='__main__': unittest.main()
