"""Exact original metric definitions on five equally sized task groups."""
from fractions import Fraction as F
def metrics(rates):
    assert set(rates)=={(s,t) for s in range(1,6) for t in range(5*s)}
    group={(s,j):sum((rates[s,t] for t in range(5*(j-1),5*j)),F(0))/5
           for s in range(1,6) for j in range(1,s+1)}
    fwt=sum((group[j,j] for j in range(1,6)),F(0))/5
    auc=sum((sum((group[s,j] for s in range(j,6)),F(0))/(6-j) for j in range(1,6)),F(0))/5
    nbt=sum((sum((group[j,j]-group[s,j] for s in range(j+1,6)),F(0))/(5-j)
             for j in range(1,5)),F(0))/4
    old=sum((rates[5,t] for t in range(20)),F(0))/20
    final=sum((rates[5,t] for t in range(25)),F(0))/25
    return {k:dict(fraction="%d/%d"%(v.numerator,v.denominator),percent=float(v*100))
            for k,v in dict(AUC=auc,FWT=fwt,NBT=nbt,OldFinal=old,FinalAll=final).items()}
