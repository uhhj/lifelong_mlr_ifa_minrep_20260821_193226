"""Evaluate every mixture component and decompose physical endpoint error, with frozen weights."""
from inputs import *
from math import factorial
def contributions(m,cmd,pred):
    masks=np.array([[(bits>>i)&1 for i in range(6)] for bits in range(64)],bool)
    coords=xyz(m,np.where(masks,pred[:6],cmd[:6]))
    a=np.zeros((6,3))
    for bits,mask in enumerate(masks):
        size=int(mask.sum())
        for j in range(6):
            if not mask[j]:
                w=factorial(size)*factorial(5-size)/factorial(6)
                a[j]+=w*(coords[bits|1<<j]-coords[bits])
    assert np.max(abs(a.sum(0)-(coords[-1]-coords[0])))<1e-6
    return a
@torch.inference_mode()
def run():
    train,m,_=init();out={}
    for group in ('regression','archive'):
        ref=read(last/('windows_'+group+'.json'))
        twins=read(outdir/('paired_'+group+'.json'))
        for e,rec in subjects(group):
            wanted=[w for w in ref[e]['windows'] if w['grid'] and w['phase']!='other']
            if not wanted:continue
            loader,seq,_,_=episode_arrays(rec,max(w['t'] for w in wanted)+9)
            archived={r['t']:r for r in twins[e]['windows'] if r['grid']}
            results=[]
            for w in wanted:
                t=w['t'];b=loader.batch(torch.tensor([t],device='cuda'))
                obs=b['obs'];anchor=b['action'][:,7]
                norm=train.normalizer['action'];a=norm.normalize(anchor)
                c,s=train._encode(obs);rates,logw,means,widths=train._side(a,s,c);K=means.shape[1]
                v=train.integrate(a.expand(K,-1),s.expand(K,-1),c.expand(K,-1),8,step_offset=rates*means[0,:,None,:])
                plans=norm.unnormalize(v).cpu().numpy();q0=np.asarray(rp.at(seq[t],'feedback'))
                endpoints=xyz(m,plans[:,-1])-xyz(m,q0)
                z=train.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0,-1].cpu().numpy()
                terms=train.loss_terms(b)
                path=train.integrate(a,s,c,8)
                expected=norm.normalize(b['action'])[:,8:16]
                explicit=(path.float()-expected.float()).square().mean()/train.vel_scale**2
                loss_error=abs(float(terms['path'])-float(explicit))
                assert loss_error<1e-4
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lowprec=train.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0,-1]
                target=np.asarray(rp.at(seq[t+7],'target'))
                attr=contributions(m,target,z)
                stored=archived[t]['variants']
                errs=[min(np.linalg.norm(endpoints-stored[n]['end'],axis=1)) for n in ('component_a','component_b')]
                err0=np.linalg.norm(xyz(m,z)-xyz(m,q0)-stored['train_zero']['end'])
                assert max([err0]+errs)<.06,(e,t,err0,errs)
                probs=logw.exp()[0].cpu().numpy().astype('float64');probs/=probs.sum()
                span={}
                for k,val in [('state',train.normalizer['agent_pos'].normalize(obs['agent_pos'])),('anchor',a),('labels',norm.normalize(b['action'][:,8:]))]:
                    span[k]=torch.amax(val.abs().reshape(-1,7),dim=0).cpu().tolist()
                results.append(dict(t=t,weights=probs.tolist(),ends=endpoints.round(6).tolist(),axes=attr.round(6).tolist(),mode_recheck=max(errs),zero_recheck=err0,span=span,
                    loss_error=loss_error,bf16_end=(output_xyz(m,lowprec)-xyz(m,q0)).round(6).tolist()))
            out[e]=results
            if len(out)%16==0:print('mixtures and axes',len(out),flush=True)
    write('components.json',out)
