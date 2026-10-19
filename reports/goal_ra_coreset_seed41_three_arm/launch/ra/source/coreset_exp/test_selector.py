import unittest,random,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('kc',Path(__file__).resolve().parents[1]/'libero/lifelong/selectors/greedy_kcenter.py')
kc=importlib.util.module_from_spec(spec);spec.loader.exec_module(kc)
class Tests(unittest.TestCase):
    def setUp(self):
        g=np.random.RandomState(17);self.v=g.normal(size=(15,8,2,6));self.a=g.normal(size=(15,7));self.ids=[f'{i:03}.pt' for i in range(15)]
    def select(self,k=7,block=3):return kc.select_arrays(self.ids,self.v,self.a,self.a,k,42,block)
    def test_quota_unique(self):self.assertEqual(len(set(self.select()[0])),7)
    def test_all(self):self.assertEqual(set(self.select(15)[0]),set(self.ids))
    def test_permutation(self):
        b=kc.select_arrays(self.ids[::-1],self.v[::-1],self.a[::-1],self.a,7,42,3)
        self.assertEqual(self.select()[0],b[0])
    def test_block(self):self.assertEqual(self.select(block=1)[0],self.select(block=50)[0])
    def test_rng(self):
        p=random.getstate();n=np.random.get_state();self.select();self.assertEqual(p,random.getstate());self.assertTrue(np.array_equal(n[1],np.random.get_state()[1]))
    def test_monotone(self):self.assertTrue(np.all(np.diff(self.select()[1]['covering_radius'])<=1e-10))
    def test_greedy_oracle(self):
        chosen,r=self.select();d=kc.Distance(self.v,self.a,self.a);matrix=d.distances(range(15));selected=[self.ids.index(chosen[0])]
        for x in chosen[1:]:
            nearest=matrix[:,selected].min(axis=1);nearest[selected]=-np.inf
            self.assertEqual(self.ids[int(nearest.argmax())],x);selected.append(self.ids.index(x))
    def test_ties(self):
        v=np.ones_like(self.v);a=np.zeros_like(self.a);ids,_=kc.select_arrays(self.ids,v,a,a,7)
        self.assertEqual(ids,self.ids[:7])
    def test_scale_constant(self):self.assertTrue(np.array_equal(kc.robust_scale(np.zeros((8,7))),np.ones(7)))
    def test_bad_k(self):
        with self.assertRaises(AssertionError):self.select(16)
    def test_bad_feature(self):
        self.v[:]=np.nan
        with self.assertRaises(AssertionError):self.select()
    def test_pairs(self):
        p=kc.reference_pairs(5);self.assertEqual(len(p),10);self.assertTrue((p[:,0]<p[:,1]).all());self.assertEqual(len(kc.reference_pairs(100)),4096)
if __name__=='__main__':unittest.main()
