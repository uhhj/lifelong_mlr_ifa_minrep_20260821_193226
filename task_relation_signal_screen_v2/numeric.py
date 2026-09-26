"""Label-blind fixed numerical definitions; no policy mutation."""
import math
import numpy as np
from scipy.spatial.distance import cdist

def grouped(v, ids):
    return np.array([np.mean(np.asarray(v)[np.asarray(ids)==i],axis=0) for i in sorted(set(ids))])

def avg(v, ids):return float(np.mean(grouped(v,ids)))

def tail(v, ids, upper=True):
    g=np.sort(grouped(v,ids));n=max(1,math.ceil(len(g)*.2))
    return float(np.mean(g[-n:] if upper else g[:n]))

def standardize(train, test, floor=1e-8):
    mu=train.mean(0);sd=np.maximum(train.std(0),floor)
    return (train-mu)/sd,(test-mu)/sd

def linear_error(x,y,halves,ids):
    errors=[]
    for side in (0,1):
        tr=halves==side;te=~tr
        a,b=standardize(x[tr],x[te]);c,d=standardize(y[tr],y[te],.05)
        # mean over n and output dimensions, alpha=1, intercept unpenalized.
        gram=a@a.T+len(a)*y.shape[1]*np.eye(len(a))
        pred=b@a.T@np.linalg.solve(gram,c)
        errors.append(avg(np.mean((pred-d)**2,axis=1),np.asarray(ids)[te]))
    return errors

def mmd(x,y,hx,hy):
    out=[]
    for side in (0,1):
        tr=np.concatenate([x[hx==side],y[hy==side]])
        _,xx=standardize(tr,x[hx!=side]);_,yy=standardize(tr,y[hy!=side])
        tt,_=standardize(tr,tr);dist=cdist(tt,tt,'sqeuclidean')
        values=dist[np.triu_indices(len(tt),1)];bw=float(np.median(values[values>0])) if np.any(values>0) else 1.
        kernel=lambda a,b:np.exp(-cdist(a,b,'sqeuclidean')/(2*bw))
        # Biased nonnegative empirical MMD; same estimator for both halves.
        out.append(float(kernel(xx,xx).mean()+kernel(yy,yy).mean()-2*kernel(xx,yy).mean()))
    return out

def conditional(x,y,ax,ay,hx,hy,action_scale):
    result=[]
    for side in (0,1):
        a=x[hx==side]; b=x[hx!=side]; c=y[hy!=side]
        aa=ax[hx==side]/action_scale;ab=ax[hx!=side]/action_scale;ac=ay[hy!=side]/action_scale
        a,b=standardize(a,b);_,c=standardize(x[hx==side],c)
        ds=cdist(b,a);dt=cdist(c,a)
        ni=np.argsort(ds,axis=1,kind='stable')[:,:5];nj=np.argsort(dt,axis=1,kind='stable')[:,:5]
        selfdist=np.take_along_axis(ds,ni,axis=1).mean(1)
        threshold=np.quantile(selfdist,.95)
        covered=np.take_along_axis(dt,nj,axis=1).mean(1)<=threshold
        coverage=float(covered.mean())
        delta_self=(ab[:,None,:]-aa[ni])**2
        delta_target=(ac[:,None,:]-aa[nj])**2
        if coverage<.1:err=grip=None
        else:
            err=float(delta_target[covered,:,:6].mean()-delta_self[:,:,:6].mean())
            grip=float(delta_target[covered,:,6].mean()-delta_self[:,:,6].mean())
        result.append([coverage,err,grip])
    return result

class Transform:
    def __init__(self,features,registry):self.features=features;self.registry=registry
    def fit(self,rows):
        self.spec=[]
        for f in self.features:
            v=np.array([r.get(f,np.nan) if r.get(f) is not None else np.nan for r in rows],float)
            good=np.isfinite(v);reg=self.registry[f];bound=reg.get('bound')
            if bound:
                scale=max(abs(a) for a in bound);z=v/scale
            else:
                nonzero=np.abs(v[good & (v!=0)])
                scale=max(float(np.median(nonzero)) if len(nonzero) else 0,reg['scale_floor'])
                z=np.arcsinh(v/scale)
            mu=float(z[good].mean()) if good.any() else 0.
            sd=float(z[good].std()) if good.any() else 0.
            zero=not good.any() or float(np.ptp(v[good]))<=reg['constant_floor'] or sd<1e-8
            self.spec.append(dict(feature=f,scale=scale,mean=mu,std=sd,zero=zero,low=float(v[good].min()) if good.any() else None,high=float(v[good].max()) if good.any() else None,bound=bound))
        return self
    def apply(self,rows):
        cols=[];audit=[]
        for a in self.spec:
            v=np.array([r.get(a['feature'],np.nan) if r.get(a['feature']) is not None else np.nan for r in rows],float)
            finite=np.isfinite(v)
            z=v/a['scale'] if a['bound'] else np.arcsinh(v/a['scale'])
            z=np.zeros(len(v)) if a['zero'] else (z-a['mean'])/a['std']
            missing=~finite;z[missing]=0
            cols.append(np.clip(z,-5,5))
            ood=int(np.sum(finite & ((v<a['low']) | (v>a['high'])))) if a['low'] is not None else 0
            audit.append(dict(feature=a['feature'],missing=int(missing.sum()),saturated=int(np.sum(abs(z)>5)),ood=ood))
        return np.column_stack(cols),audit

def ridge(train,test,y,features,registry,alpha=1.):
    transform=Transform(features,registry).fit(train)
    x,_=transform.apply(train);z,audit=transform.apply(test)
    mu=float(np.mean(y));sd=float(np.std(y))
    yc=(y-mu)/sd if sd>1e-12 else np.zeros(len(y))
    center=x.mean(0);xc=x-center
    coef=np.linalg.solve(xc.T@xc/len(x)+alpha*np.eye(x.shape[1]),xc.T@yc/len(x))
    raw=((z-center)@coef)*sd+mu
    return np.clip(raw,-1,1),raw,dict(transform=transform.spec,center=center.tolist(),coef=coef.tolist(),y_mean=mu,y_std=sd,alpha=alpha,audit=audit)
