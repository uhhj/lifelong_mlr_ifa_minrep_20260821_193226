"""Pure synthetic selector verification; not evidence of experimental performance."""
import unittest
import numpy as np
from selection import prepare,repair,describe,search,brute_search


class SelectionTest(unittest.TestCase):
    def test_exact_search_against_direct_enumeration(self):
        rng=np.random.RandomState(42)
        for trial in range(100):
            n=12;ids=[f'{i:03d}.pt' for i in range(n)]
            rows={name:dict(demo_id=f'd{i//6}',target=i%6,length=6,
                            group=int(rng.choice([0,2]))) for i,name in enumerate(ids)}
            a=rng.normal(size=(n,3));D=np.linalg.norm(a[:,None]-a[None,:],axis=-1)
            if trial%2:D+=rng.uniform(0,.01,(n,n));np.fill_diagonal(D,0)
            original=[ids[i] for i in rng.choice(n,6,replace=False)]
            result=repair(ids,rows,original,D,verify_brute=True)
            self.assertLessEqual(result['after']['radius'],result['before']['radius']+1e-12)
            self.assertGreaterEqual(result['after']['unique_times'],result['before']['unique_times'])

    def test_duplicate_targets_reference_and_endpoint(self):
        ids=['a','b','c','d'];rows={
            'a':dict(demo_id='d0',target=0,length=4,group=0),
            'b':dict(demo_id='d0',target=3,length=4,group=14),
            'c':dict(demo_id='d0',target=3,length=4,group=14),
            'd':dict(demo_id='d0',target=2,length=4,group=2)}
        m=prepare(ids,rows)
        self.assertEqual(m['U'],3);self.assertEqual(m['groups'][1],11)
        D=np.array([[0,1,1,.5],[1,0,0,.5],[1,0,0,.5],[.5,.5,.5,0]])
        r=repair(ids,rows,['a','b'],D,verify_brute=True)
        self.assertIn('d0',r['after']['endpoint_demos'])
        self.assertNotIn('c',r['selected_ids'])

    def test_noop_preserves_order(self):
        ids=['a','b','c'];rows={s:dict(demo_id='d',target=i,length=3,group=2) for i,s in enumerate(ids)}
        D=np.eye(3)*-1+1
        r=repair(ids,rows,['c','b','a'],D,verify_brute=True)
        self.assertEqual(r['selected_ids'],['c','b','a']);self.assertEqual(r['swaps'],[])

    def test_mean_floor_is_independent_of_radius(self):
        # Run exact comparisons on arbitrary nonmetric distances with many equal radii.
        rng=np.random.RandomState(12)
        for _ in range(50):
            ids=[str(i) for i in range(8)]
            rows={s:dict(demo_id='d',target=i,length=8,group=2*(i%2)) for i,s in enumerate(ids)}
            D=rng.randint(0,4,(8,8)).astype(float);np.fill_diagonal(D,0)
            repair(ids,rows,['0','1','2','7'],D,verify_brute=True)

    def test_identity_conflict_rejected(self):
        rows={'a':dict(demo_id='d',target=1,length=3,group=0),
              'b':dict(demo_id='d',target=1,length=3,group=2)}
        with self.assertRaises(AssertionError):prepare(['a','b'],rows)


if __name__=='__main__':unittest.main(verbosity=2)
