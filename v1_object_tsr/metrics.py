from fractions import Fraction

def exact_metrics(matrix):
    r=[[Fraction(str(x)) if x is not None else None for x in row] for row in matrix]
    n=len(r)
    fwt=sum(r[j][j] for j in range(n))/n
    nbt=sum(sum(r[j][j]-r[i][j] for i in range(j+1,n))/(n-j-1) for j in range(n-1))/(n-1)
    auc=sum(sum(r[i][j] for i in range(j,n))/(n-j) for j in range(n))/n
    return dict(auc=auc,fwt=fwt,nbt=nbt)


def floats(values):return {k:float(v) for k,v in values.items()}



def build_result(seed,matrix,baseline):
    values=exact_metrics(matrix);base=exact_metrics(baseline)
    return dict(seed=seed,r=matrix,metrics=floats(values),baseline_metrics=floats(base),
        exact_metrics={k:str(v) for k,v in values.items()},
        delta=floats({k:values[k]-base[k] for k in values}),
        automatic_additional_seeds=False)

