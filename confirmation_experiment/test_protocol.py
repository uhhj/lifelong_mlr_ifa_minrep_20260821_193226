import unittest
from confirmation_experiment.protocol import derived_signals,snapshot_slot,decide

class ProtocolTests(unittest.TestCase):
    def signal(self,g,r,nn=1,no=1):
        m=dict(has_reference=bool(no),tiny_reference=r==0,mixed_grad_norm=abs(g),reference_grad_norm=abs(r),dot_before=g*r)
        return derived_signals(m,nn,no)
    def test_ordinary_conflict_cancelled_by_replay(self):
        s=self.signal(.25,1)
        self.assertEqual(s['n_dot_r'],-.5);self.assertEqual(s['z'],.5)
    def test_strong_conflict(self):
        s=self.signal(-.5,1)
        self.assertEqual(s['n_dot_r'],-2);self.assertEqual(s['z'],2)
        self.assertEqual(s['cos_new_reference'],-1)
    def test_no_conflict(self):self.assertLessEqual(self.signal(2,1)['z'],0)
    def test_no_old_no_new_zero_reference(self):
        for nn,no,r in ((10,0,0),(0,10,1),(5,5,0)):
            self.assertIsNone(self.signal(1,r,nn,no)['z'])
    def test_zero_new_norm_explicit(self):
        s=self.signal(.5,1)
        self.assertTrue(s['new_norm_cancellation']);self.assertIsNone(s['cos_new_reference'])
    def test_snapshot_fixed_budget(self):
        used=set()
        for t in (6,7,8,9):
            for e in range(100):
                for trigger,z in ((True,2),(False,.5),(False,-1)):
                    k=snapshot_slot(t,e,trigger,z,used,100)
                    if k:used.add(k)
        self.assertEqual(len(used),16)
    def test_snapshot_no_substitution_and_future20(self):
        self.assertIsNone(snapshot_slot(7,35,False,-1,set(),100))
        self.assertIsNone(snapshot_slot(7,35,True,2,set(),19))
        self.assertIsNone(snapshot_slot(7,20,True,2,set(),100))
    def deltas(self,auc,fwt=0,final=0):return [dict(auc=x,fwt=fwt,final_mean_success=final) for x in auc]
    def test_not_supported_primary_nonpositive(self):
        self.assertTrue(decide(self.deltas([-1,1,0]),{}).endswith('NOT_SUPPORTED'))
    def test_not_supported_systematic_plasticity_cost(self):
        self.assertTrue(decide(self.deltas([2,2,2],-3),{}).endswith('NOT_SUPPORTED'))
    def test_inconclusive_small_dominant_or_old_harm(self):
        for ds,olds in ((self.deltas([.1,.2,.3]),{}),(self.deltas([10,.1,.1]),{}),
                        (self.deltas([2,2,2]),{6:[-5,-5,-5]})):
            self.assertTrue(decide(ds,olds).endswith('INCONCLUSIVE'))
    def test_supported(self):self.assertTrue(decide(self.deltas([2,2,1]),{}).endswith('_SUPPORTED'))
if __name__=='__main__':unittest.main()
