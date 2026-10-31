"""Training-only probes and full-state branch selection; no environment access."""
import contextlib,math
from runtime import *
from support_memory import weights,from_selection,retention
from legacy_losses import cost

@contextlib.contextmanager
def auxiliary(a):
    rr=rng();modes={n:m.training for n,m in a.policy.named_modules()};buffers={n:v.detach().clone() for n,v in a.policy.named_buffers()}
    a.policy.eval()
    try:yield
    finally:
        assert all(torch.equal(v,buffers[n]) for n,v in a.policy.named_buffers())
        for n,m in a.policy.named_modules():m.training=modes[n]
        restore_rng(rr)

def setup_probes(ds,memory):
    stage=ds.stage
    ids=[i for i,r in enumerate(ds.rows) if r['task']==stage and r['bc_valid']]
    assert ids
    g=np.random.RandomState(20261027+SEED*100+stage);chosen=sorted(int(x) for x in g.choice(ids,min(64,len(ids)),replace=False))
    y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']).float() for i in ids])
    sc=torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]]).cuda()
    old={}
    for t,m in memory.items():
        gen=np.random.RandomState(20261027+SEED*1000+t)
        old[t]=sorted(int(x) for x in gen.choice(len(m['rows']),min(8,len(m['rows'])),replace=False)) if m['rows'] else []
    meta=dict(new_indices=chosen,old_indices=old,new_scale=sc.tolist(),support=weights(memory,'Q'),support_trajectories={str(t):m['support_trajectories'] for t,m in memory.items()},new_population=len(ids),source='TRAINING_ONLY_FIXED_NOT_VALIDATION')
    return dict(ds=ds,new=chosen,old=old,scale=sc,metadata=meta)

def new_terms(a,x,scale):
    hh=hidden(a.policy.module,x['feature']);bc,valid=masked_bc(a.policy.module,hh,x,True)
    assert valid.all()
    mu,sd,z=a.policy.module.policy_head.forward_fn(hh);k=z.detach().argmax(-1);ix=torch.arange(len(k),device=k.device)
    ex=cost(mu,sd,x['next_action'],scale)[ix,k].mean()
    return bc,ex

@torch.no_grad()
def probe(a,pr,memory):
    with auxiliary(a):
        bc=0.;ex=0.;calls=0;count=0
        for start in range(0,len(pr['new']),10):
            ids=pr['new'][start:start+10];batch=torch.utils.data.default_collate([pr['ds'][i] for i in ids]);x=a.map_tensor_to_device(batch[0])
            b,e=new_terms(a,x,pr['scale']);bc+=float(b)*len(ids);ex+=float(e)*len(ids);count+=len(ids);calls+=1
        old=0.;contrib={};ww=weights(memory,'Q')
        for t,ids in pr['old'].items():
            if not ids:contrib[str(t)]=0.;continue
            rx,rt,ct=from_selection(a.policy.module,memory,[(t,i) for i in ids],ww,len(ids)*len(memory))
            v=float(rx+.1*rt);old+=v;contrib[str(t)]=v;calls+=ct['forward_calls']
        out=dict(bc_new=bc/count,exec_new=ex/count,retain_support=old,retain_by_task=contrib,probe_windows=count+sum(len(x) for x in pr['old'].values()),policy_calls=calls)
        assert all(math.isfinite(out[k]) for k in ['bc_new','exec_new','retain_support'])
        return out

def choose(c,q):
    for x in [c,q]:assert all(math.isfinite(x[k]) for k in ['bc_new','exec_new','retain_support'])
    checks=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
    return ('Q' if all(checks.values()) else 'C'),checks

def directional_diagnostic(a,batch,memory,pr,arm):
    stage=pr['ds'].stage
    tasks=[int(t) for t in batch[1]];ids=[i for i,t in enumerate(tasks) if t==stage and bool(batch[0]['causal_bc_valid'][i])]
    if not ids:return None
    before=capture(a)
    with auxiliary(a):
        allx=a.map_tensor_to_device(batch[0]);x={k:(v[ids] if torch.is_tensor(v) else v) for k,v in allx.items()}
        bc,ex=new_terms(a,x,pr['scale']);rx,rt,ct=retention(a.policy.module,memory,stage,a.step,arm if arm!='C' else 'Q')
        re=.1*(rx+.1*rt);pars=[p for p in a.policy.parameters() if p.requires_grad]
        gs=[torch.autograd.grad(a.loss_scale*v,pars,retain_graph=True,allow_unused=True) for v in [bc,ex,re]]
        def norm(g):return sum(float(v.double().square().sum()) for v in g if v is not None)**.5
        def pair(g,h):
            dot=sum(float(x.double().mul(y.double()).sum()) for x,y in zip(g,h) if x is not None and y is not None);den=norm(g)*norm(h)
            return dict(dot=dot,cosine=dot/den if den else None)
        out=dict(bc_new=float(bc.detach()),exec_new=float(ex.detach()),retain_weighted=float(re.detach()),norms=dict(bc_new=norm(gs[0]),exec_new=norm(gs[1]),retain=norm(gs[2])),retain_vs_bc=pair(gs[2],gs[0]),retain_vs_exec=pair(gs[2],gs[1]),contributions=ct['contributions'],current_valid=len(ids),extra_autograd_calls=3,retain_counterfactual_only=arm=='C')
    assert equal(before,capture(a));return out
