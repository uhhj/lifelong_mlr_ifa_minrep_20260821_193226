"""Pure synthetic CPU tests; no claims about real LIBERO success."""
import random
import unittest
import numpy as np
from selector_reference import select_anchor_repair, stable_slots, anchor_ids, _deletion_scores

class Tests(unittest.TestCase):
    def test_uniform_draw_matches_parent_random_prefix(self):
        ids = ['%03d.pt' % i for i in range(21)]
        for seed in (42,43,44):
            self.assertEqual(anchor_ids(ids,11,seed),random.Random(seed).sample(ids,11)[:5])

    def test_inclusion_and_minimum_swaps(self):
        rnd=np.random.RandomState(4)
        for n,q in [(2,1),(8,4),(21,11),(38,20)]:
            ids = ['%03d.pt' % i for i in range(n)]
            a = rnd.normal(size=(n,4)); D=np.sqrt(np.sum((a[:,None]-a[None])**2,axis=-1));np.fill_diagonal(D,0)
            centers=ids[::max(1,n//q)][:q]
            out,rep=select_anchor_repair(ids,centers,D,seed=42)
            self.assertEqual(len(out),q)
            self.assertTrue(set(anchor_ids(ids,q,42)).issubset(out))
            self.assertEqual(rep['swap_count'],len(set(rep['anchor_ids'])-set(centers)))

    def test_exact_deletion_scores_against_brute(self):
        rnd=np.random.RandomState(11)
        for n,q in [(6,3),(17,8),(30,12)]:
            x=rnd.normal(size=(n,3));D=np.sqrt(np.sum((x[:,None]-x[None])**2,axis=-1))
            for _ in range(10):
                aug=list(rnd.choice(n,size=q,replace=False))
                r,m=_deletion_scores(D,aug)
                for i in range(q):
                    chosen=aug[:i]+aug[i+1:]
                    v=D[:,chosen].min(axis=1)
                    self.assertAlmostEqual(r[i],v.max(),places=11)
                    self.assertAlmostEqual(m[i],v.mean(),places=11)

    def test_exact_self_parity(self):
        ids=['%03d.pt'%i for i in range(10)]
        D=np.abs(np.arange(10)[:,None]-np.arange(10)[None,:]).astype(float)
        parent=ids[:5]
        a=anchor_ids(ids,5,42)
        # A design requiring anchors already inside S0 should return same list.
        eligible=[x for x in ids if x not in a]
        centers=a+eligible[:3]
        result,report=select_anchor_repair(ids,centers,D,seed=42)
        self.assertEqual(result,centers)
        self.assertEqual(report['swap_count'],0)

    def test_reject_invalid_parent_random(self):
        ids=['%03d.pt'%i for i in range(10)]
        D=np.abs(np.arange(10)[:,None]-np.arange(10)[None,:]).astype(float)
        with self.assertRaises(ValueError):
            select_anchor_repair(ids,ids[:5],D,seed=42,expected_random=ids[-5:])

    def test_slot_alignment_invariant(self):
        old=['t6__%03d.pt'%i for i in range(10)]
        new=old[:2]+old[3:7]+old[8:]+['t6__010.pt','t6__011.pt']
        result,changes=stable_slots(old,new)
        self.assertEqual(len(changes),2)
        self.assertEqual(result[2],'t6__010.pt')
        self.assertEqual(result[7],'t6__011.pt')
        for i,x in enumerate(old):
            if x in new:self.assertEqual(x,result[i])
        self.assertEqual(stable_slots(old,old),(old,[]))

    def test_exact_batch_invariance_outside_changed_slots(self):
        old=['t6__%03d.pt'%i for i in range(40)]
        changed=old[1:]+['t6__077.pt']
        new,changes=stable_slots(old,changed)
        changed_slots={c['slot'] for c in changes}
        old_mix=old+['current_%03d.pt'%i for i in range(30)]
        new_mix=new+['current_%03d.pt'%i for i in range(30)]
        for epoch in range(20):
            perm=list(range(len(old_mix))); random.Random(42+epoch).shuffle(perm)
            for i in perm:
                if i not in changed_slots:self.assertEqual(old_mix[i],new_mix[i])
        self.assertEqual(len(changes),1)

    def test_no_global_rng_consumption(self):
        before=random.getstate()
        ids=['%03d.pt'%i for i in range(12)]
        D=np.abs(np.arange(12)[:,None]-np.arange(12)[None,:]).astype(float)
        select_anchor_repair(ids,ids[:6],D,seed=42)
        self.assertEqual(before,random.getstate())

if __name__=='__main__':unittest.main(verbosity=2)
