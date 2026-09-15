import unittest,tempfile
import numpy as np
from .common import trial_summary,round_seed
class Tests(unittest.TestCase):
    def test_warmup_not_success(self):
        flags=np.ones((5,10),bool)
        self.assertFalse(any(x['success'] for x in trial_summary(flags,[-4,-3,-2,-1,0])))
    def test_success_accumulates_and_first(self):
        a=np.zeros((4,10),bool);a[1,2]=True;a[3,2]=True;a[2,3]=True
        r=trial_summary(a,[1,2,3,4]);self.assertEqual(sum(x['success'] for x in r),2);self.assertEqual(r[2]['first_success_step'],2)
    def test_seeds_unique_and_arm_independent(self):
        v=[round_seed(i,t,r) for i in range(36) for t in range(6,9) for r in range(2)]
        self.assertEqual(len(set(v)),len(v))
    def test_balancing_not_trial_pooling(self):
        # An origin with three tasks must not outweigh one with two tasks.
        vals=[[1,0],[0,0,0]]
        self.assertEqual(np.mean([np.mean(x) for x in vals]),.25)
        self.assertNotEqual(np.mean(sum(vals,[])),.25)
    def test_pair_sign(self):
        mlr=np.array([0,1,0,1]);qp=np.array([1,0,0,1])
        self.assertEqual(int((qp-mlr).sum()),int(((qp==1)&(mlr==0)).sum()-((qp==0)&(mlr==1)).sum()))
    def test_missing_pair_fails(self):
        from .analyze import paired
        with self.assertRaises(AssertionError):paired([{'point':'x','task':6,'arm':'MLR','trial':0,'success':1}])
if __name__=='__main__':unittest.main()
