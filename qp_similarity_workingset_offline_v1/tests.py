import unittest
import numpy as np
from .grouping import group
from .engine import run
from .data import synthetic
from .reference import compare
class Tests(unittest.TestCase):
    def test_safe_zero_all_branches(self):
        A=np.eye(3);b=np.array([-1.,0.,-.2]);v=np.arange(3)
        for arm,seed in [('FULL_WS',None),('SIM_ADD',None),('RANDOM_ADD',0)]:
            r=run(A,b,v,arm,random_seed=seed,k_design=3)
            self.assertEqual(r['meta']['status'],'PASS');self.assertFalse(r['x'].any())
    def test_representative_miss_requires_reinsert(self):
        N=np.array([[1.,1.],[-.001,.001]]);N/=np.linalg.norm(N,axis=0)
        A=N.T@N;b=N.T@np.array([0.,1.])
        r=run(A,b,np.arange(2),'SIM_ADD')
        self.assertEqual(r['meta']['k0'],1);self.assertFalse(r['first']['full_kkt_pass'])
        self.assertEqual(r['trace'][0]['added'],[1]);self.assertTrue(r['meta']['full_kkt_pass'])
        self.assertEqual(r['meta']['k_final'],2)
    def test_opposite_not_merged(self):
        groups,_=group(np.array([[1.,-1.],[-1.,1.]]),np.arange(2),.999)
        self.assertEqual(len(groups),2)
    def test_chain_not_transitive(self):
        angle=np.arccos(.999)*.8*np.arange(3);N=np.stack([np.cos(angle),np.sin(angle)])
        groups,_=group(N.T@N,np.arange(3),.999)
        self.assertEqual(groups,[[0,1],[2]])
    def test_duplicate_certified_or_explicit_unresolved(self):
        A=np.ones((3,3));b=np.ones(3)*.5
        for arm in ('FULL_WS','SIM_ADD','RANDOM_ADD'):
            r=run(A,b,np.arange(3),arm,random_seed=0,k_design=1)
            if r['meta']['status']=='PASS':self.assertLessEqual(r['meta']['float64']['kkt'],1e-8)
            else:self.assertTrue('UNRESOLVED' in r['meta']['status'] or 'SINGULAR' in r['meta']['status'])
    def test_valid_order_and_medoid_tie(self):
        groups,reps=group(np.ones((3,3)),np.array([9,2,5]),.999)
        self.assertEqual(groups,[[1,2,0]]);self.assertEqual(reps,[1])
    def test_no_input_mutation(self):
        A=np.array([[1.,.9999],[.9999,1.]]);b=np.array([.5,.4]);a=A.copy();c=b.copy()
        run(A,b,np.arange(2),'SIM_ADD');np.testing.assert_array_equal(A,a);np.testing.assert_array_equal(b,c)
    def test_synthetic_structure(self):
        for kind in ('low','high','mixed'):
            z=synthetic(32,kind,2026092201)
            np.testing.assert_allclose(z['A'],z['N'].T@z['N']);np.testing.assert_allclose(z['b'],z['N'].T@z['v'])
            self.assertAlmostEqual(np.linalg.norm(z['v']),1.)
    def test_update_measure_nonunique_dual(self):
        r=compare(np.ones((2,2)),np.ones(2),np.array([1.,0.]),np.array([0.,1.]),N=np.array([[1.,1.]]))
        self.assertEqual(r['E_update'],0.);self.assertTrue(r['update_budget_pass'])
    def test_material_negative_q_is_error(self):
        r=compare(np.array([[-1.]]),np.ones(1),np.ones(1),np.zeros(1))
        self.assertEqual(r['status'],'NEGATIVE_UPDATE_QUADRATIC')
if __name__=='__main__':unittest.main()
