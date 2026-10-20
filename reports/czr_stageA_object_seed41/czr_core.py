"""Deterministic nonparametric repair; no optimizer or simulator dependency."""
import numpy as np

class Memory:
    def __init__(self,z,parent,label):
        z=np.asarray(z,dtype=np.float64)
        self.mu=z.mean(0)
        p=self.phi(z)
        errors=parent!=label
        if not (~errors).any():
            raise ValueError("No correct examples for protection")
        self.centers=p[errors]
        self.parent=parent[errors].copy()
        self.label=label[errors].copy()
        self.indices=np.flatnonzero(errors)
        self.radius=np.array([.45*np.linalg.norm(p[~errors]-x,axis=1).min()
                              for x in self.centers])
    def phi(self,z):
        a=np.asarray(z,dtype=np.float64)-self.mu
        return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
    def query(self,z,parent):
        z=self.phi(z)
        out=np.asarray(parent).copy()
        trigger=np.zeros(len(z),dtype=bool)
        conflicts=np.zeros(len(z),dtype=bool)
        counts=np.zeros(len(z),dtype=int)
        for j,x in enumerate(z):
            hit=(np.linalg.norm(self.centers-x,axis=1)<self.radius)&(self.parent==parent[j])
            labels=np.unique(self.label[hit]);counts[j]=hit.sum()
            if len(labels)>1:conflicts[j]=True
            elif len(labels)==1:out[j]=labels[0];trigger[j]=True
        return out,trigger,conflicts,counts

def language_route(e,canonical):
    e=np.asarray(e,dtype=np.float64);canonical=np.asarray(canonical,dtype=np.float64)
    n=np.linalg.norm(e,axis=-1,keepdims=True)
    canonical=canonical/np.linalg.norm(canonical,axis=-1,keepdims=True)
    hit=np.max(np.abs(e[:,None]/np.maximum(n[:,None],1e-12)-canonical[None]),axis=-1)<=1e-6
    return (n[:,0]>1e-12)&(hit.sum(-1)==1)&hit[:,6]

def metric(parent,candidate,label):
    a=np.asarray(parent)==label;b=np.asarray(candidate)==label
    ss=int((a&b).sum());sf=int((a&~b).sum());fs=int((~a&b).sum());ff=int((~a&~b).sum())
    return dict(n=len(a),SS=ss,SF=sf,FS=fs,FF=ff,parent_correct=ss+sf,
                candidate_correct=ss+fs,net=fs-sf)

def properties():
    z=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    p=np.array([-1,-1,-1,-1]);y=np.array([1,-1,-1,-1])
    m=Memory(z,p,y)
    q,t,_,_=m.query(z,p)
    assert np.array_equal(q,y) and t.tolist()==[True,False,False,False]
    assert not m.query(z,np.ones(4))[1].any()
    empty=Memory(z,p,p);assert not empty.query(z,p)[1].any()
    zero=Memory(np.ones((2,2)),np.array([-1,-1]),np.array([1,-1]))
    assert not zero.query(np.ones((2,2)),np.array([-1,-1]))[1].any()
    m.centers=np.repeat(m.centers,2,axis=0);m.parent=np.repeat(m.parent,2)
    m.radius=np.repeat(m.radius,2);m.label=np.array([-1,1])
    assert m.query(z[:1],p[:1])[2][0] and not m.query(z[:1],p[:1])[1][0]
    canon=np.eye(7,8);e=np.concatenate([canon,np.zeros((1,8)),np.ones((1,8))])
    assert language_route(e,canon).tolist()==[False]*6+[True,False,False]
    return dict(status="PASS",properties=6,synthetic_only=True)
