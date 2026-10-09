"""Single-native-graph decoder gradient gate. No optimizer or RNG operations."""
import math
import time
import torch


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def gradient_statistics(a,b):
    device=next(x.device for x in a+b if x is not None)
    sums=torch.zeros(3,dtype=torch.float64,device=device)
    for ga,gb in zip(a,b):
        for g in (ga,gb):
            if g is not None: assert bool(torch.isfinite(g).all()), 'NONFINITE_PROBE_GRADIENT'
        count=(ga if ga is not None else gb)
        if count is None: continue
        for start in range(0,count.numel(),262144):
            x=ga.reshape(-1)[start:start+262144].double() if ga is not None else None
            y=gb.reshape(-1)[start:start+262144].double() if gb is not None else None
            if x is not None: sums[1]+=x.square().sum()
            if y is not None: sums[2]+=y.square().sum()
            if x is not None and y is not None: sums[0]+=(x*y).sum()
    dot,aa,bb=sums.tolist()
    assert all(math.isfinite(v) for v in (dot,aa,bb)), 'NONFINITE_GATE_REDUCTION'
    return dict(dot=dot,bc_decoder_norm=math.sqrt(aa),ifa_decoder_norm=math.sqrt(bb),
        cosine=dot/math.sqrt(aa*bb) if aa and bb else None,
        bc_unused=sum(x is None for x in a),ifa_unused=sum(x is None for x in b))


def select_loss(bc,native,terms,params,force=None):
    assert params and force in (None,'open','closed')
    assert all(bool(torch.isfinite(x)) for x in [bc,native]+terms), 'NONFINITE_LOSS'
    values=[float(t.detach()) for t in terms]
    row=dict(weighted_terms=values,has_ifa=bool(terms),hinge_zero=bool(terms) and all(v==0 for v in values),
        w=None,reason='NO_IFA',dot=None,bc_decoder_norm=None,ifa_decoder_norm=None,cosine=None,
        bc_unused=None,ifa_unused=None,probe_seconds=0.,dot_seconds=0.,retained_ifa=sum(values))
    if not terms:return native,row
    if row['hinge_zero']:
        row['reason']='HINGE_ZERO';return native,row
    aux=terms[0]
    for term in terms[1:]:aux=aux+term
    sync(bc.device);begin=time.monotonic()
    a=torch.autograd.grad(bc,params,retain_graph=True,create_graph=False,allow_unused=True)
    b=torch.autograd.grad(aux,params,retain_graph=True,create_graph=False,allow_unused=True)
    sync(bc.device);row['probe_seconds']=time.monotonic()-begin;begin=time.monotonic()
    assert any(x is not None for x in b), 'POSITIVE_IFA_DISCONNECTED_FROM_DECODER'
    row.update(gradient_statistics(a,b))
    sync(bc.device);row['dot_seconds']=time.monotonic()-begin
    del a,b,aux
    row['w']=int(row['dot']>=0)
    row['reason']='KEEP' if row['w'] else 'NEGATIVE_DOT'
    if row['bc_decoder_norm']==0 or row['ifa_decoder_norm']==0:row['reason']='ZERO_GRADIENT'
    if force is not None:row['w']=int(force=='open');row['reason']='ENGINEERING_FORCE_'+force.upper()
    row['retained_ifa']=row['w']*sum(values)
    return (native if row['w'] else bc),row
