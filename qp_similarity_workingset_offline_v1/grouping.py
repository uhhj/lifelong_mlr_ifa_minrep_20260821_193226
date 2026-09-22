import numpy as np

def group(A,valid,tau):
    S=(A+A.T)/2.
    assert np.max(np.abs(S))<=1+1e-12, 'Material cosine normalization violation'
    S=np.clip(S,-1.,1.)  # grouping only; original QP A is read-only
    order=sorted(range(len(valid)),key=lambda i:int(valid[i]));groups=[]
    for i in order:
        eligible=[(float(S[i,g].mean()),j) for j,g in enumerate(groups) if np.all(S[i,g]>=tau)]
        if eligible:
            gid=min(eligible,key=lambda p:(-p[0],p[1]))[1];groups[gid].append(i)
        else:groups.append([i])
    representatives=[]
    for g in groups:
        if len(g)==1:representatives.append(g[0]);continue
        means={i:float(np.mean([S[i,j] for j in g if j!=i])) for i in g}
        representatives.append(min(g,key=lambda i:(-means[i],int(valid[i]))))
    return groups,sorted(representatives,key=lambda i:int(valid[i]))
