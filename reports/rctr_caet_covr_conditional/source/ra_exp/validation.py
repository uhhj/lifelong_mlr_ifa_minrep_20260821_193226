"""Numeric witnesses use the original FP64 tolerance; identities remain exact."""
import json
def same_witness(a,b):
    a=json.loads(json.dumps(a));b=json.loads(json.dumps(b))
    def check(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:check(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for u,v in zip(x,y):check(u,v)
        elif isinstance(x,float):assert abs(x-y)<=1e-12
        else:assert x==y
    check(a,b)
