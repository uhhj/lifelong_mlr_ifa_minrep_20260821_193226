import unittest,itertools
import numpy as np
from sc_exp.selection import group_id,quotas,repair

class Tests(unittest.TestCase):
    def test_time_and_sign(self):
        self.assertEqual([group_id(t,9,-1)//3 for t in range(9)],[0,0,1,1,2,2,3,3,4])
        self.assertEqual([group_id(0,1,g) for g in [-1,0,1]],[12,13,14])
    def test_quota_exhaustive(self):
        rng=np.random.RandomState(4)
        for n in range(2,18):
            g=rng.randint(0,5,n).tolist()
            for k in range(1,n+1):
                s=rng.choice(n,k,False).tolist();q,r=quotas(g,s)
                allq=[v for v in itertools.product(*[range(r['lower'][x],r['upper'][x]+1) for x in sorted(q)]) if sum(v)==k]
                def key(v):return (sum(abs(gsum-vv) for gsum,vv in zip([r['original'].get(x,0) for x in sorted(q)],v)),sum(abs(vv*n-k*g.count(x)) for x,vv in zip(sorted(q),v)),tuple(-vv for vv in v))
                self.assertEqual(tuple(q[x] for x in sorted(q)),min(allq,key=key))
    def test_swap_against_bruteforce(self):
        rng=np.random.RandomState(17)
        for repeat in range(30):
            n=20;ids=['%03d'%x for x in range(n)];g=rng.randint(0,6,n).tolist();z=rng.randn(n,4);D=np.linalg.norm(z[:,None]-z[None,:],axis=2)
            original=rng.choice(n,10,False).tolist();result,r=repair(ids,g,[ids[i] for i in original],D)
            chosen=original[:];q=r['target']
            for sw in r['swaps']:
                deficits=[i for i in range(n) if i not in chosen and sum(g[j]==g[i] for j in chosen)<q[g[i]]]
                near=D[:,chosen].min(1);mx=max(near[i] for i in deficits);x=min(i for i in deficits if near[i]>=mx-1e-12)
                excess=[j for j in chosen if sum(g[i]==g[j] for i in chosen)>q[g[j]]]
                costs={j:D[:,[i for i in chosen+[x] if i!=j]].min(1).max() for j in excess};mn=min(costs.values());y=min(j for j in excess if costs[j]<=mn+1e-12)
                self.assertEqual((sw['added'],sw['removed']),(ids[x],ids[y]));chosen[chosen.index(y)]=x
            self.assertEqual([ids[i] for i in chosen],result)
    def test_noop_and_ties(self):
        ids=list('abcdefgh');g=[0]*4+[1]*4;D=np.ones((8,8))-np.eye(8)
        self.assertEqual(repair(ids,g,['b','a','f','e'],D)[0],['b','a','f','e'])
        x,r=repair(ids,g,list('abcd'),D);self.assertEqual(x,['e','f','c','d']);self.assertEqual(r['minimum_swaps'],2)
    def test_rare_zero_quota(self):
        ids=list('abcdef');g=[0]*5+[1];D=np.ones((6,6))-np.eye(6)
        self.assertEqual(repair(ids,g,['a','b'],D)[0],['a','b'])
if __name__=='__main__':unittest.main()
