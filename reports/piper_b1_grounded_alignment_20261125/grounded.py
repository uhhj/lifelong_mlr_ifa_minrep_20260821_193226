"""Archive-only B1 planar alignment experiment."""
from pathlib import Path
import json
import sys
import os

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='2'
import numpy as np


def setup(work):
    diag=work/'offline_diagnostics'
    sys.path[:0]=[str(diag/'b1_feedback_xy_20261125'),str(diag/'b1_spatial_ablation_20261121_2034')]
    import policy_xy as xy
    rel,prior=xy.setup(work)
    return xy,rel,prior


def inspect(work,out):
    print('batches:',[p.name for p in (work/'batches').iterdir()],flush=True)
    print('archive roots:',[p.name for p in work.iterdir() if any(k in p.name.lower() for k in ('select','episode','capture','record','demo','dataset','b1'))],flush=True)
    for d in (work/'batches').glob('B1*'):
        print(d.name,[(p.name,p.is_dir()) for p in list(d.iterdir())[:12]],flush=True)


def robust_center(points):
    pts=np.array(points,dtype=float)
    if not len(pts):return None,0
    neighbors=np.linalg.norm(pts[:,None]-pts[None,:],axis=-1)<=10.
    j=np.argmax(neighbors.sum(1))
    return np.median(pts[neighbors[j]],axis=0),int(neighbors[j].sum())


class NativeBlock:
    """Locate one stationary block in native overhead images; training references are private."""
    def __init__(self,bank,bounds):
        self.bank=bank
        self.bounds=bounds

    def estimate(self,images):
        import cv2
        from block_track import detect
        points=[]
        for im in images:
            point=detect(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            if point is not None:points.append(point)
        p,count=robust_center(points)
        if count>=3:return p,'color'
        x0,y0,x1,y1=self.bounds
        maps=[]
        for im in images[::2]:
            roi=im[y0:y1,x0:x1]
            scores=np.stack([cv2.matchTemplate(roi,tmp,cv2.TM_CCOEFF_NORMED) for tmp in self.bank])
            # At least three different reference appearances should agree at the same position.
            maps.append(np.partition(scores,-3,axis=0)[-3:].mean(0))
        score=np.median(maps,axis=0)
        v,u=np.unravel_index(np.argmax(score),score.shape)
        if score[v,u]<.40:
            return (p,'single_color') if count else (None,'missing')
        half=self.bank.shape[1]//2
        return np.array([x0+u+half,y0+v+half],dtype=float),'template'


class Heatmap:
    """A translation-equivariant block-center heatmap on a fixed native-resolution crop."""
    bounds=(128,255,352,447)
    def __init__(self,path=None,device='cpu'):
        import torch
        from torch import nn
        self.device=device
        blocks=[]
        for a,b,k,s,dil in [(3,16,5,2,1),(16,24,3,2,1),(24,32,3,1,1),(32,48,3,1,2),
                              (48,48,3,1,2),(48,32,3,1,1)]:
            blocks.extend([nn.Conv2d(a,b,k,stride=s,padding=dil*(k//2),dilation=dil),
                           nn.GroupNorm(4,b),nn.SiLU()])
        self.net=nn.Sequential(*blocks,nn.Conv2d(32,1,1)).to(device)
        if path:self.net.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        self.net.eval()

    def crop(self,image):
        x,y,X,Y=self.bounds
        return image[y:Y,x:X]

    def positions(self,scores):
        import torch
        b,_,h,w=scores.shape
        flat=scores.reshape(b,-1)
        arg=flat.argmax(1)
        gy,gx=torch.meshgrid(torch.arange(h,device=scores.device),torch.arange(w,device=scores.device),indexing='ij')
        local=(gx[None]-arg[:,None,None]%w).abs()<=2
        local&=(gy[None]-arg[:,None,None]//w).abs()<=2
        probs=flat.masked_fill(~local.flatten(1),-1e6).softmax(1)
        px=(probs*gx.flatten()).sum(1)*4+self.bounds[0]
        py=(probs*gy.flatten()).sum(1)*4+self.bounds[1]
        return torch.stack([px,py],1)

    def estimate(self,images):
        import torch
        x=torch.from_numpy(np.stack([self.crop(im) for im in images])).permute(0,3,1,2).to(self.device).float()/255
        with torch.inference_mode():
            logits=self.net(x)
            # Stationary block during approach, common center over the eight causal observations.
            point=self.positions(logits.mean(0,keepdim=True))[0].cpu().numpy()
        return point,'heatmap'


def train_heatmap(work,out,augmented=False):
    xy,rel,_=setup(work)
    from annotate import images
    import cv2,torch
    import torch.nn.functional as F
    cv2.setNumThreads(1); torch.set_num_threads(2)
    records,geo,_,fk=rel.data(work)
    train=[r for r in records if r['train']]
    assert len(train)==50
    np_rng=np.random.default_rng(2711)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    crop=Heatmap().crop
    frames=[]; labels=[]; members=[]; phase=[]; backgrounds=[]; cutouts=[]
    polygons={32:[[23,18],[54,18],[57,24],[56,59],[21,59],[20,54]],
              52:[[26,16],[57,18],[59,24],[56,54],[29,51],[22,46]]}
    for i,r in enumerate(train):
        ep=r['episode']; cut=min(k for k in (r['close'],r['contact']) if k is not None)
        ticks=sorted({r['frame'],*np.arange(0,cut-3,3).tolist(),cut-18})
        fs=images(r['folder'],'overhead')[r['first']:]
        if augmented:
            backgrounds.append(crop(cv2.imread(str(fs[-1]))).copy())
            if r['index'] in polygons:
                u,v=np.rint(r['uv']).astype(int)
                ref=cv2.imread(str(fs[r['frame']]))[v-32:v+48,u-40:u+40]
                mask=np.zeros((80,80),np.uint8)
                cv2.fillConvexPoly(mask,np.array(polygons[r['index']],np.int32),255)
                cutouts.append((i,ref.copy(),cv2.GaussianBlur(mask,(3,3),.6)))
        for t in ticks:
            if t<0 or t>=cut:continue
            frames.append(crop(cv2.imread(str(fs[t]))).copy())
            labels.append(r['uv']);members.append(i);phase.append(t==cut-18)
    pix=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).contiguous().to(device)
    uv=torch.as_tensor(np.array(labels),dtype=torch.float32,device=device)
    members=np.array(members);phase=np.array(phase)
    # Choose epoch count only from a fixed training-only validation split.
    validation=set(np_rng.permutation(50)[:10].tolist())
    history=[]
    best=0
    def run(learn,valid,budget,final=False):
        nonlocal best
        torch.manual_seed(437);torch.cuda.manual_seed_all(437)
        rng=np.random.default_rng(437)
        det=Heatmap(device=device)
        net=det.net.train()
        if augmented:
            library=[(torch.from_numpy(p).permute(2,0,1).to(device).float()/255,
                      torch.from_numpy(m).to(device).float()/255) for i,p,m in cutouts if i in set(members[learn])]
            bg=torch.from_numpy(np.stack(backgrounds)).permute(0,3,1,2).to(device)
            assert len(library)>0
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.005)
        h,w=pix.shape[-2:]
        yy,xx=torch.meshgrid(torch.arange(h//4,device=device)*4,torch.arange(w//4,device=device)*4,indexing='ij')
        for step in range(1,budget+1):
            ids=rng.choice(learn,24)
            x=pix[ids].float()/255
            points=uv[ids]-torch.tensor(det.bounds[:2],device=device)
            if augmented:
                # The same stationary table can contain a cube independently of the arm's location.
                # Keep clean full cubes as well as partially occluded examples in ordinary frames.
                for ix in range(10):
                    x[ix]=bg[members[ids[ix]]].float()/255
                    item,alpha=library[rng.integers(len(library))]
                    u=int(rng.integers(50,171));v=int(rng.integers(56,133))
                    brightness=rng.uniform(.68,1.18)
                    box=x[ix,:,v-32:v+48,u-40:u+40]
                    box.copy_(box*(1-alpha)+item*brightness*alpha)
                    points[ix]=torch.tensor([u,v],device=device)
            # Translation, mirroring and photometric changes share the same target transform.
            dx,dy=rng.integers(-20,21,size=2).tolist()
            x=torch.roll(x,(dy,dx),(-2,-1));points=points+torch.tensor([dx,dy],device=device)
            if rng.random()<.5:
                x=x.flip([-1]);points[:,0]=w-1-points[:,0]
            scale=.78+.44*torch.rand((len(ids),3,1,1),device=device)
            x=(x*scale+(torch.rand((len(ids),1,1,1),device=device)-.5)*.12).clamp(0,1)
            sq=(xx-points[:,0,None,None])**2+(yy-points[:,1,None,None])**2
            target=(-sq/32.).flatten(1).softmax(1)
            logits=net(x).flatten(1)
            loss=-(target*F.log_softmax(logits,1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if not final and step in (150,300,500,750):
                net.eval()
                with torch.no_grad():
                    pred=det.positions(net(pix[valid].float()/255))
                    err=torch.linalg.norm(pred-uv[valid],dim=1)
                    metric=err.mean().item()
                    history.append(dict(step=step,mean_px=metric,median_px=err.median().item()))
                print('training-only validation',step,metric,flush=True)
                if metric<=min(r['mean_px'] for r in history):best=step
                net.train()
        if final:torch.save(net.cpu().state_dict(),out/'private_heatmap.pt')
    run(np.flatnonzero(~np.isin(members,list(validation))),np.flatnonzero(np.isin(members,list(validation))&phase),750)
    run(np.arange(len(members)),None,best,True)
    rel.dump(out/'inspection.json',dict(heatmap_steps=best,train_frames=len(frames),groups=50,object_paste_training=augmented,
        validation_groups=[train[i]['episode'] for i in sorted(validation)],validation=history,
        heatmap_sha256=rel.digest(out/'private_heatmap.pt')))
    print('native heatmap frozen',len(frames),best,flush=True)


def fit(work,out):
    _,rel,_=setup(work)
    import cv2
    from annotate import images
    records,geo,_,fk=rel.data(work)
    train=[r for r in records if r['train']]
    assert len(train)==50
    tip=np.asarray(geo['tip_mm'])
    matrix=np.asarray(geo['table_map'])
    bank=[]
    uv=np.stack([r['uv'] for r in train])
    for r in train:
        img=cv2.imread(str(images(r['folder'],'overhead')[r['first']+r['frame']]))
        u,v=np.rint(r['uv']).astype(int)
        bank.append(img[v-19:v+20,u-19:u+20])
    bank=np.stack(bank)
    lo=np.maximum(np.floor(uv.min(0)-60),[0,0]).astype(int)
    hi=np.minimum(np.ceil(uv.max(0)+61),[640,480]).astype(int)
    exps=work/'offline_diagnostics/b1_alignment_windows_20261125'
    selected=json.loads((exps/'selection.json').read_bytes())
    fractions=[]
    true_x=[]; screen_x=[]; mid_x=[]; targets=[]; groups=[]
    reader_path=work/'offline_diagnostics/b1_feedback_xy_20261125/private_xy_policy.npz'
    oldfit=json.loads((reader_path.parent/'fit.json').read_bytes())
    assert rel.digest(reader_path)==oldfit['policy_sha256']
    with np.load(reader_path) as readers:
        w,b=readers['wood_w'],readers['wood_b']
    with np.load(exps/'private_selected.npz') as s:
        cache_eps=s['episodes']; cache_off=s['offsets']; ft=s['features'].astype(float)
    cap_samples=[]
    for r in train:
        ep=r['episode']; cut=min(k for k in (r['close'],r['contact']) if k is not None)
        goal=fk(r['q'][r['contact']])
        rows=[z for z in selected['episodes'][ep]['qualifying'] if z['selected']]
        files=images(r['folder'],'overhead')[r['first']:]
        native=NativeBlock(bank[[a['episode']!=ep for a in train]],np.r_[lo,hi])
        for z in rows:
            t=z['t']; state=fk(r['q'][t])
            move=fk(r['q'][t+8])-state
            offset=goal-state
            ratio=move@offset/max(1.,offset@offset)
            fractions.append(ratio)
            fidx=np.flatnonzero((cache_eps==ep)&(cache_off==cut-t))
            assert len(fidx)==1
            mids=ft[fidx[0]]@w+b
            frames=[cv2.imread(str(f)) for f in files[t-7:t+1]]
            native_uv,_=native.estimate(frames)
            if native_uv is None:continue
            true_x.append(offset);screen_x.append(np.r_[native_uv,1.]@matrix-state)
            mid_x.append(np.r_[mids,1.]@matrix-state)
            groups.append(ep);targets.append(move)
        j=(r['close'] if r['close'] is not None else r['contact'])-18
        cap_samples.append(np.linalg.norm(fk(r['q'][j+8])-fk(r['q'][j])))
    gain=float(np.clip(np.median(fractions),0.05,1.0))
    cap=float(np.percentile(cap_samples,75))
    y=np.array(targets); eps=np.array(groups); coeff={}
    for key,x in [('oracle',true_x),('native',screen_x),('mid',mid_x)]:
        x=np.array(x)
        losses=np.zeros(len(rel.PENALTIES))
        for val in rel.folds(set(groups)):
            v=np.isin(eps,sorted(val))
            for i,p in enumerate(rel.PENALTIES):
                h=rel.fit_head(x[~v],y[~v],eps[~v],p)
                losses[i]+=rel.weights(eps[v])@np.linalg.norm(rel.residual(h,x[v])-y[v],axis=1)
        h=rel.fit_head(x,y,eps,rel.PENALTIES[int(np.argmin(losses))])
        coeff.update({key+'_'+k:v for k,v in h.items()})
    np.savez(out/'private_policy.npz',bank=bank,bounds=np.r_[lo,hi],matrix=matrix,tip=tip,
        gain=gain,cap=cap,reader_w=w,reader_b=b,**coeff)
    rel.dump(out/'fit.json',dict(train_episodes=sorted(r['episode'] for r in train),gain=gain,
        cap_mm=cap,n_fit=len(y),n_selected=len(fractions),policy_sha256=rel.digest(out/'private_policy.npz'),
        protocol_sha256=rel.digest(out/'protocol.md'),reader_sha256=oldfit['policy_sha256']))
    print('fit complete',len(y),gain,cap,flush=True)


def move(data,start,goal,mode,source):
    if goal is None:return start
    dist=goal-start
    if mode=='feedback':
        v=float(data['gain'])*dist
    else:
        key='oracle' if mode=='oracle_trained' else source
        h={k:data[key+'_'+k] for k in ('cy','cx','sx','slope')}
        v=h['cy']+((dist-h['cx'])/h['sx'])@h['slope']
    return start+v*min(1.,float(data['cap'])/max(1e-12,np.linalg.norm(v)))


def pose_features(q,flange,tip):
    joints=np.asarray(q)[:6]
    T=flange(np.rad2deg(joints.astype(float)))
    point=T[:3,3]+T[:3,:3]@tip
    return np.r_[point,T[:3,:3].flatten(),np.sin(joints),np.cos(joints)]


def fit_registration(work,out):
    """Calibrate the pose-dependent difference between image-based and expert contact-center coordinates."""
    import cv2,torch
    from scipy.linalg import solve
    xy,rel,_=setup(work)
    from fit_gripper import flange
    from annotate import images
    torch.set_num_threads(2);cv2.setNumThreads(1)
    records,_,_,fk=rel.data(work)
    train=[r for r in records if r['train']]
    with np.load(out/'private_policy.npz') as arch:par=dict(arch)
    detector=Heatmap(out/'private_heatmap.pt')
    rows=[]; ys=[]; names=[]
    for r in train:
        stop=min(t for t in [r['close'],r['contact']] if t is not None)
        imgs=images(r['folder'],'overhead')[r['first']:]
        true=fk(r['q'][r['contact']])
        for t in sorted({stop-k for k in (36,30,24,18,12,8) if stop-k>=7}):
            p,_=detector.estimate([cv2.imread(str(f)) for f in imgs[t-7:t+1]])
            projected=np.r_[p,1.]@par['matrix']
            rows.append(pose_features(r['q'][t],flange,par['tip']))
            ys.append(true-projected);names.append(r['episode'])
    X,Y,G=np.stack(rows),np.stack(ys),np.array(names)
    def fit_one(t,lam):
        w=rel.weights(G[t]);w/=w.sum()
        mu=w@X[t];sd=np.sqrt(w@((X[t]-mu)**2)).clip(1e-3)
        z=(X[t]-mu)/sd;center=w@Y[t]
        slopes=solve(z.T@(w[:,None]*z)+lam*np.eye(z.shape[1]),z.T@(w[:,None]*(Y[t]-center)),assume_a='pos')
        return mu,sd,center,slopes
    regs=np.array([.003,.01,.03,.1,.3,1.,3.,10.])
    error=np.zeros(len(regs));neutral=0.
    for val in rel.folds(set(names)):
        v=np.isin(G,sorted(val))
        w=rel.weights(G[v])
        neutral+=w@np.linalg.norm(Y[v],axis=1)
        for i,lam in enumerate(regs):
            mu,sd,beta,slope=fit_one(~v,lam)
            err=beta+((X[v]-mu)/sd)@slope-Y[v]
            error[i]+=w@np.linalg.norm(err,axis=1)
    best=int(np.argmin(error))
    mu,sd,beta,slope=fit_one(np.ones(len(G),bool),regs[best])
    limit=float(np.percentile(np.linalg.norm(Y,axis=1),90))
    np.savez(out/'private_registration.npz',mu=mu,sd=sd,beta=beta,slope=slope,limit=limit)
    info=json.loads((out/'inspection.json').read_bytes())
    info['registration']=dict(fit_windows=len(names),train_cv_uncorrected_mm=neutral/50,
             train_cv_corrected_mm=error[best]/50,penalty=float(regs[best]),clip_mm=limit,
             sha256=rel.digest(out/'private_registration.npz'))
    rel.dump(out/'inspection.json',info)
    print('held-out-training calibration error',neutral/50,error[best]/50,flush=True)


def registered(uv,q,par,reg,flange):
    goal=np.r_[uv,1.]@par['matrix']
    feat=pose_features(q,flange,par['tip'])
    pred=reg['beta']+((feat-reg['mu'])/reg['sd'])@reg['slope']
    limited=pred*min(1.,float(reg['limit'])/max(1e-9,np.linalg.norm(pred)))
    # Reduce pose extrapolation beyond the final approach region.
    dist=np.linalg.norm(goal-feat[:2])
    blend=np.clip((65.-dist)/30.,0.,1.)
    return goal+blend*limited


def context_features(qhistory,wrist,flange,tip):
    import cv2
    poses=[pose_features(q,flange,tip) for q in (qhistory[0],qhistory[-1])]
    state=np.r_[poses[1],poses[1][:3]-poses[0][:3],qhistory[-1,6],qhistory[-1,:6]-qhistory[0,:6]]
    image=cv2.resize(wrist,(20,20),interpolation=cv2.INTER_AREA).astype(float).ravel()/255
    return state,image


def fit_fusion(work,out):
    """Train a camera-position gain and a wrist/pose residual against independent contact feedback."""
    import torch,cv2
    xy,rel,_=setup(work)
    from annotate import images
    from fit_gripper import flange
    torch.set_num_threads(2);cv2.setNumThreads(1)
    records,geo,_,fk=rel.data(work)
    train=[r for r in records if r['train']]
    with np.load(out/'private_policy.npz') as a:par=dict(a)
    hm=Heatmap(out/'private_heatmap.pt')
    xs=[];ws=[];native=[];start=[];truth=[];episodes=[];offsets=[]
    for r in train:
        stop=min(v for v in (r['close'],r['contact']) if v is not None)
        overhead=images(r['folder'],'overhead')[r['first']:]
        wrist=images(r['folder'],'wrist')[r['first']:]
        for gap in (36,30,24,18,12,8):
            t=stop-gap
            if t<7:continue
            uv,_=hm.estimate([cv2.imread(str(f)) for f in overhead[t-7:t+1]])
            st,im=context_features(r['q'][t-7:t+1],cv2.imread(str(wrist[t])),flange,par['tip'])
            xs.append(st);ws.append(im);start.append(fk(r['q'][t]))
            native.append(np.r_[uv,1.]@par['matrix']);truth.append(fk(r['q'][r['contact']]))
            episodes.append(r['episode']);offsets.append(gap)
    X,W,P,N,Y=map(np.stack,(xs,ws,start,native,truth))
    E=np.array(episodes); offsets=np.array(offsets)
    gains=[.35,.5,.7,1.]
    mixes=[0.,.25,1.,4.]
    penalties=[.0005,.003,.015,.08,.4,2.]
    err=np.zeros((len(gains),len(mixes),len(penalties)))
    def design(ix):
        xm=X[ix].mean(0); xsd=X[ix].std(0).clip(1e-3)
        xx=(X-xm)/xsd/np.sqrt(X.shape[1])
        wm=W[ix].mean(0); wsd=np.sqrt(np.sum((W[ix]-wm)**2)/sum(ix)).clip(1e-6)
        ww=(W-wm)/wsd
        return xm,xsd,xx,wm,wsd,ww
    for val in rel.folds(set(E)):
        v=np.isin(E,sorted(val)); t=~v
        _,_,xx,_,_,ww=design(t)
        for mi,mix in enumerate(mixes):
            k=xx@xx[t].T+mix*(ww@ww[t].T)
            eig,V=np.linalg.eigh(k[t])
            left=k[v]@V
            for gi,gain in enumerate(gains):
                prior=P+gain*(N-P); residual=Y-prior
                b=residual[t].mean(0);rhs=V.T@(residual[t]-b)
                for ri,reg in enumerate(penalties):
                    pred=prior[v]+b+left@(rhs/(np.maximum(eig,0)[:,None]+reg*sum(t)))
                    errors=np.linalg.norm(pred-Y[v],axis=1)
                    err[gi,mi,ri]+=.5*errors.mean()+.5*errors[offsets[v]==18].mean()
    gi,mi,ri=np.unravel_index(np.argmin(err),err.shape)
    gain,mix,reg=gains[gi],mixes[mi],penalties[ri]
    xm,xsd,xx,wm,wsd,ww=design(np.ones(len(E),bool))
    raw=Y-P-gain*(N-P)
    b=raw.mean(0); kernel=xx@xx.T+mix*(ww@ww.T)
    alpha=np.linalg.solve(kernel+reg*len(E)*np.eye(len(E)),raw-b)
    lim=float(np.percentile(np.linalg.norm(raw,axis=1),95))
    np.savez(out/'private_fusion.npz',xm=xm,xsd=xsd,wm=wm,wsd=wsd,state_w=xx.T@alpha,wrist_w=mix*ww.T@alpha,
        b=b,gain=gain,limit=lim)
    info=json.loads((out/'inspection.json').read_bytes())
    info['fusion']=dict(gain=gain,mix=mix,reg=reg,training_contact_cv_mm=float(err.min()/5),windows=len(E),
        sha256=rel.digest(out/'private_fusion.npz'),gain_grid=gains,cv_best_by_gain_mm=(err.min((1,2))/5).tolist())
    rel.dump(out/'inspection.json',info)
    print('training only fusion',info['fusion'],flush=True)


def fused(uv,qhistory,wrist,par,fusion,flange):
    state,image=context_features(qhistory,wrist,flange,par['tip'])
    raw=np.r_[uv,1.]@par['matrix']
    pos=state[:2]
    x=(state-fusion['xm'])/fusion['xsd']/np.sqrt(len(state))
    w=(image-fusion['wm'])/fusion['wsd']
    shift=fusion['b']+x@fusion['state_w']+w@fusion['wrist_w']
    shift*=min(1.,float(fusion['limit'])/max(np.linalg.norm(shift),1e-12))
    near=pos+float(fusion['gain'])*(raw-pos)+shift
    blend=np.clip((65-np.linalg.norm(raw-pos))/30,0.,1.)
    return blend*near+(1-blend)*raw


def soft(v,radius):
    return v*np.maximum(0.,1.-radius/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9))


def close_weight(height,contact_z,cue,params):
    z0,ramp,quiet,span=params
    vertical=np.clip((height-contact_z-z0)/ramp,0.,1.)
    displaced=np.clip((np.linalg.norm(cue,axis=-1)-quiet)/span,0.,1.)
    return np.maximum(vertical,displaced)


def settle_goal(uv,qh,wrist,par,s,flange):
    state,im=context_features(qh,wrist,flange,par['tip'])
    pos=state[:2]
    x=(state-s['xm'])/s['xsd']/np.sqrt(len(state));w=(im-s['wm'])/s['wsd']
    inferred=s['b']+x@s['wx']+w@s['ww']
    raw=np.r_[uv,1.]@par['matrix']
    cue=raw-pos-inferred[2:]
    local=pos+soft(inferred[:2],s['local_radius'])+s['gain']*soft(cue,s['cue_radius'])
    blend=np.clip((65-np.linalg.norm(raw-pos))/30,0.,1.)
    goal=blend*local+(1-blend)*raw
    if 'guard' in s:
        weight=close_weight(state[2],s['contact_z'],cue,s['guard'])
        goal=pos+weight*(goal-pos)
    return goal


def settle_fit(work,out):
    """Use group-cross-fitted wrist/pose expectations to learn when small corrections should stop."""
    import cv2,torch
    xy,rel,_=setup(work)
    from fit_gripper import flange
    from annotate import images
    torch.set_num_threads(2);cv2.setNumThreads(1)
    records,_,_,fk=rel.data(work)
    train=[r for r in records if r['train']]
    adapter=B1OfflineAlignment(work,out)
    meta=json.loads((out/'inspection.json').read_bytes())
    mix=meta['fusion']['mix']; lam=meta['fusion']['reg']
    states=[];ims=[];raw=[];centers=[];short=[];ids=[];gaps=[];contact_z={}
    for r in train:
        cut=min(v for v in (r['close'],r['contact']) if v is not None)
        contact_z[r['episode']]=pose_features(r['q'][r['contact']],flange,adapter.par['tip'])[2]
        overhead=images(r['folder'],'overhead')[r['first']:]
        wrist=images(r['folder'],'wrist')[r['first']:]
        for gap in (36,30,24,18,12,8,4):
            t=cut-gap
            if t<7:continue
            qh=r['q'][t-7:t+1]; pos=fk(qh[-1])
            st,im=context_features(qh,cv2.imread(str(wrist[t])),flange,adapter.par['tip'])
            uv,_=adapter.det.estimate([cv2.imread(str(f)) for f in overhead[t-7:t+1]])
            states.append(st);ims.append(im);raw.append(np.r_[uv,1.]@adapter.par['matrix']-pos)
            centers.append(fk(r['q'][r['contact']])-pos)
            short.append(fk(r['q'][min(t+8,r['contact'])])-pos)
            ids.append(r['episode']);gaps.append(gap)
    X,W,A,C,D=map(np.stack,(states,ims,raw,centers,short))
    E=np.array(ids);gaps=np.array(gaps);Y=np.c_[C,A]
    def regress(mask):
        xm=X[mask].mean(0);xsd=X[mask].std(0).clip(1e-3)
        xx=(X-xm)/xsd/np.sqrt(X.shape[1])
        wm=W[mask].mean(0);wsd=np.sqrt(np.sum((W[mask]-wm)**2)/sum(mask)).clip(1e-6)
        ww=(W-wm)/wsd
        k=xx[mask]@xx[mask].T+mix*(ww[mask]@ww[mask].T)
        b=Y[mask].mean(0)
        a=np.linalg.solve(k+lam*sum(mask)*np.eye(sum(mask)),Y[mask]-b)
        wx=xx[mask].T@a;wi=mix*ww[mask].T@a
        return b+xx@wx+ww@wi,dict(xm=xm,xsd=xsd,wm=wm,wsd=wsd,b=b,wx=wx,ww=wi)
    oof=np.empty_like(Y)
    for val in rel.folds(set(ids)):
        valid=np.isin(E,sorted(val))
        oof[valid]=regress(~valid)[0][valid]
    gain0=float(adapter.par['gain']);cap=float(adapter.par['cap'])
    def step(v):
        v=gain0*v
        return v*np.minimum(1.,cap/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9))
    # Match the expert on natural histories; also demand a response to a known 25-mm target translation.
    u=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],axis=1)
    dist=lambda z:np.linalg.norm(z,axis=-1)
    blend=np.clip((65-dist(A))/30,0.,1.)[:,None]
    residual=A-oof[:,2:]
    candidates=[]
    for local in (0.,2.,4.,6.):
        for cue in (0.,3.,6.,9.,12.):
            for gain in (.35,.5,.7,1.):
                local_move=soft(oof[:,:2],local)
                near=local_move+gain*soft(residual,cue)
                pred=step(blend*near+(1-blend)*A)
                e=dist(pred-D)
                loss=.50*e.mean()+.25*e[gaps<=8].mean()+.25*dist(pred-C)[gaps<=18].mean()
                shift=25*u[None,:,:]
                alt=A[:,None]+shift
                altblend=np.clip((65-dist(alt))/30,0.,1.)[:,:,None]
                altered=step(altblend*(local_move[:,None]+gain*soft(residual[:,None]+shift,cue))+(1-altblend)*alt)
                signed=((altered-pred[:,None])*u).sum(-1)
                robust=np.mean(signed>1.)
                if robust>=.85:candidates.append((loss,local,cue,gain,robust))
    best=min(candidates)
    _,params=regress(np.ones(len(E),bool))
    _,params['local_radius'],params['cue_radius'],params['gain'],robust=best
    # Estimate the work surface from independent training episodes. Decrease small motions
    # at grasp height, while a sufficiently novel block location re-opens the correction.
    frozen_z=np.empty(len(E))
    for val in rel.folds(set(ids)):
        frozen_z[np.isin(E,sorted(val))]=np.median([z for ep,z in contact_z.items() if ep not in val])
    local=soft(oof[:,:2],params['local_radius'])
    near=local+params['gain']*soft(residual,params['cue_radius'])
    desired=blend*near+(1-blend)*A
    shift=25*u[None,:,:]
    alt=A[:,None]+shift
    altblend=np.clip((65-dist(alt))/30,0.,1.)[:,:,None]
    shifted=altblend*(local[:,None]+params['gain']*soft(residual[:,None]+shift,params['cue_radius']))+(1-altblend)*alt
    def criteria(pred,alternative):
        err=dist(pred-D)
        obj=.4*err.mean()+.30*err[gaps<=8].mean()+.30*dist(pred-C)[gaps<=8].mean()
        return obj,np.mean(((alternative-pred[:,None])*u).sum(-1)>1.)
    plain,_=criteria(step(desired),step(shifted))
    options=[]
    for z0 in (-3.,0.,3.,6.):
        for ramp in (6.,12.,20.):
            for quiet in (6.,10.,14.,18.):
                for span in (6.,12.):
                    g=(z0,ramp,quiet,span)
                    weight=close_weight(X[:,2],frozen_z,residual,g)
                    different=close_weight(X[:,None,2],frozen_z[:,None],residual[:,None]+shift,g)
                    obj,response=criteria(step(desired*weight[:,None]),step(shifted*different[:,:,None]))
                    if response>=.85:options.append((obj,g,response))
    guarded,g,response=min(options)
    if guarded<plain:
        params['guard']=np.asarray(g)
        params['contact_z']=float(np.median(list(contact_z.values())))
    np.savez(out/'private_settle.npz',**params)
    meta['settle']=dict(local_radius=best[1],cue_radius=best[2],gain=best[3],n_train=len(E),
                        train_shift_positive_gt1=robust,sha256=rel.digest(out/'private_settle.npz'))
    meta['settle']['height_gate']=dict(parameters=list(g),applied='guard' in params,
        train_crossfit_loss_before=plain,train_crossfit_loss_after=guarded,
        train_translation_positive_gt1=response)
    rel.dump(out/'inspection.json',meta)
    print('training-only settling fit',meta['settle'],flush=True)


def evaluate(work,out,heat=False):
    xy,rel,_=setup(work)
    import cv2
    from annotate import images
    _,prior,records,geo,goals,fk,d,training,starts,target,_=xy.load_windows(work)
    fit_record=json.loads((out/'fit.json').read_bytes())
    assert fit_record['policy_sha256']==rel.digest(out/'private_policy.npz')
    with np.load(out/'private_policy.npz') as archive: par=dict(archive)
    native=NativeBlock(par['bank'],par['bounds'])
    reg=None
    if heat:
        import torch
        torch.set_num_threads(2)
        hm=Heatmap(out/'private_heatmap.pt')
        if (out/'private_registration.npz').exists():
            from fit_gripper import flange
            with np.load(out/'private_registration.npz') as arc:reg=dict(arc)
    pixel={}
    pos={ep:i for i,ep in enumerate(d['episodes']) if ep not in training}
    aim={};location=[];natural=[];directions=[]
    variants=['base','oracle/learned','mid/learned','native/learned','oracle/feedback','mid/feedback','native/feedback']
    if heat:variants+=['mid/oracle_trained','heatmap/oracle_trained','heatmap/feedback']
    if reg is not None:variants+=['registered/feedback','registered/oracle_trained']
    fusion=None;context={}
    if heat and (out/'private_fusion.npz').exists():
        from fit_gripper import flange
        with np.load(out/'private_fusion.npz') as arc:fusion=dict(arc)
        variants+=['fused/feedback']
    steady=None
    if fusion is not None and (out/'private_settle.npz').exists():
        with np.load(out/'private_settle.npz') as arc:steady=dict(arc)
        variants+=['settle/feedback']
    for ep,i in pos.items():
        r=records[ep];t=d['indices'][i]
        ff=images(r['folder'],'overhead')[r['first']+t-7:r['first']+t+1]
        images8=[cv2.imread(str(p)) for p in ff]
        detected,src=native.estimate(images8)
        mid=d['features'][i].astype(float)@par['reader_w']+par['reader_b']
        contact=fk(r['q'][r['contact']])
        aim[ep]={'oracle':contact,'native':None if detected is None else np.r_[detected,1.]@par['matrix'],
                  'mid':np.r_[mid,1.]@par['matrix']}
        location.append(dict(episode=ep,method=src,native_px=None if detected is None else float(np.linalg.norm(detected-r['uv'])),
            native_contact_mm=None if detected is None else float(np.linalg.norm(aim[ep]['native']-contact)),
            mid_px=float(np.linalg.norm(mid-r['uv'])),label_contact_mm=float(np.linalg.norm(goals[ep]-contact))))
        if heat:
            hp,_=hm.estimate(images8)
            pixel[ep]=hp
            aim[ep]['heatmap']=np.r_[hp,1.]@par['matrix']
            location[-1].update(heatmap_px=float(np.linalg.norm(hp-r['uv'])),
                heatmap_contact_mm=float(np.linalg.norm(aim[ep]['heatmap']-contact)))
            if reg is not None:
                registration=registered(hp,r['q'][t],par,reg,flange)
                location[-1]['registered_contact_mm']=float(np.linalg.norm(registration-contact))
            if fusion is not None:
                w=cv2.imread(str(images(r['folder'],'wrist')[r['first']+t]))
                context[ep]=(r['q'][t-7:t+1],w)
                fgoal=fused(hp,*context[ep],par,fusion,flange)
                location[-1]['fused_contact_mm']=float(np.linalg.norm(fgoal-contact))
    originals={}; changes={}
    def append(ep,donor,base,mode,key):
        i=pos[ep]; start=starts[i]
        if mode=='base':return fk(base[-1])
        source,law=mode.split('/')
        if source=='settle':goal=settle_goal(pixel[donor],*context[ep],par,steady,flange)
        elif source=='fused':goal=fused(pixel[donor],*context[ep],par,fusion,flange)
        elif source=='registered':goal=registered(pixel[donor],records[ep]['q'][d['indices'][i]],par,reg,flange)
        else:goal=aim[donor][source]
        end=move(par,start,goal,law,source)
        path=start+np.arange(1,9)[:,None]/8*(end-start)
        q,check=xy.joint_commands(base,path,par['tip'])
        if check['xy_max_mm']>1e-4:raise RuntimeError(('planar reconstruction',key,check))
        return fk(q[-1])
    for ep,i in pos.items():
        c=aim[ep]['oracle'];start=starts[i];expert=start+target[i,-1]
        for mode in variants:
            p=append(ep,ep,d['base'][i],mode,'natural')
            originals[ep,mode]=p
            natural.append(dict(episode=ep,mode=mode,error_mm=np.linalg.norm(p-expert),
                advance_mm=np.linalg.norm(start-goals[ep])-np.linalg.norm(p-goals[ep]),
                contact_advance_mm=np.linalg.norm(start-c)-np.linalg.norm(p-c),
                contact_end_mm=np.linalg.norm(p-c),travel_mm=np.linalg.norm(p-start)))
    for k,(ep,donor) in enumerate(zip(d['recipients'],d['donors'])):
        delta=goals[donor]-goals[ep];axis=delta/np.linalg.norm(delta)
        cdelta=aim[donor]['oracle']-aim[ep]['oracle']; caxis=cdelta/np.linalg.norm(cdelta)
        for mode in variants:
            p=append(ep,donor,d['swapped'][k],mode,'swap'); orig=originals[ep,mode]
            start=starts[pos[ep]]
            directions.append(dict(recipient=ep,donor=donor,mode=mode,signed_mm=(p-orig)@axis,
                contact_signed_mm=(p-orig)@caxis,separation_mm=np.linalg.norm(delta),
                toward_new_mm=np.linalg.norm(start-goals[donor])-np.linalg.norm(p-goals[donor]),
                toward_contact_mm=np.linalg.norm(start-aim[donor]['oracle'])-np.linalg.norm(p-aim[donor]['oracle'])))
    file=out/'outcomes.json'
    rel.dump(file,dict(pairs=geo['pairs'],locations=location,natural=natural,directions=directions))
    print('evaluated',len(directions),'directions',flush=True)


class B1OfflineAlignment:
    """Pre-grasp B1 joint-plan adapter. All inputs are ordinary arrays; no hardware API is used."""
    def __init__(self,work,out):
        self.xy,self.rel,_=setup(work)
        from fit_gripper import flange
        self.flange=flange
        fit_record=json.loads((out/'fit.json').read_bytes())
        report=json.loads((out/'inspection.json').read_bytes())
        assert self.rel.digest(out/'private_policy.npz')==fit_record['policy_sha256']
        assert self.rel.digest(out/'private_heatmap.pt')==report['heatmap_sha256']
        with np.load(out/'private_policy.npz') as archive:self.par=dict(archive)
        self.det=Heatmap(out/'private_heatmap.pt')
        self.reg=None
        if 'registration' in report:
            assert self.rel.digest(out/'private_registration.npz')==report['registration']['sha256']
            with np.load(out/'private_registration.npz') as archive:self.reg=dict(archive)
        self.fusion=None
        if 'fusion' in report:
            assert self.rel.digest(out/'private_fusion.npz')==report['fusion']['sha256']
            with np.load(out/'private_fusion.npz') as archive:self.fusion=dict(archive)
        self.steady=None
        if 'settle' in report:
            assert self.rel.digest(out/'private_settle.npz')==report['settle']['sha256']
            with np.load(out/'private_settle.npz') as archive:self.steady=dict(archive)

    def fk(self,q):
        pose=self.flange(np.rad2deg(np.asarray(q,dtype=float)[...,:6]))
        return (pose[...,:3,3]+pose[...,:3,:3]@self.par['tip'])[...,:2]

    def predict(self,images8,q,base,approach=True,calibrate=True,prior_centers=None,wrist=None,qhistory=None):
        if not approach:return np.array(base,copy=True),None
        if len(images8)!=8 or base.shape!=(8,7):raise ValueError('Expected eight frames and eight joint targets')
        if calibrate and self.steady is not None:
            if wrist is None or np.shape(qhistory)!=(8,7):
                raise ValueError('Final B1 approach requires a wrist image and eight feedback samples')
            if not np.allclose(q,qhistory[-1],rtol=0,atol=1e-6):
                raise ValueError('Latest joint feedback disagrees with the observation history')
        uv,_=self.det.estimate(images8)
        if prior_centers is not None:
            prior_centers.append(uv.copy())
            uv=np.median(prior_centers[-5:],axis=0)
        return self.plan(uv,q,base,calibrate,wrist,qhistory),uv

    def plan(self,uv,q,base,calibrate,wrist=None,qhistory=None):
        if self.steady is not None and wrist is not None and qhistory is not None and calibrate:
            goal=settle_goal(uv,qhistory,wrist,self.par,self.steady,self.flange)
        elif self.fusion is not None and wrist is not None and qhistory is not None and calibrate:
            goal=fused(uv,qhistory,wrist,self.par,self.fusion,self.flange)
        elif calibrate and self.reg is not None:goal=registered(uv,q,self.par,self.reg,self.flange)
        else:goal=np.r_[uv,1.]@self.par['matrix']
        start=self.fk(q)
        end=move(self.par,start,goal,'feedback','heatmap')
        path=start+np.arange(1,9)[:,None]/8*(end-start)
        joints,check=self.xy.joint_commands(base,path,self.par['tip'])
        if max(check['xy_max_mm'],check['height_max_mm'],check['angle_max_deg'])>1e-4:
            raise ValueError('Offline reconstruction failed')
        return joints


def field(work,out):
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    xy,rel,_=setup(work)
    from probe import load_model
    adapter=B1OfflineAlignment(work,out)
    model,_=load_model(work,'baseline')
    sessions=[
        ('20261123_234228_1446439_B1-ode50-20261123',[248,350]),
        ('20261123_234953_1448999_B1-ode50-20261123',[218,357])]
    stats={}
    for name,ref in sessions:
        root=work/'policy_runs'/name
        rows=[json.loads(l) for l in (root/'inputs.jsonl').read_bytes().splitlines()]
        by_stamp={r['stamp']:r for r in rows}
        anchors=[r for r in rows if r['mode_at_query']=='execute' and r['plan']['ready'] and r['stamp']==r['plan']['chunk_stamp']]
        table=[]
        history_centers=[]
        for r in anchors:
            history=[by_stamp[s] for s in r['history']]
            scenes=[cv2.imread(str(root/h['images']['scene'])) for h in history]
            raw=[sc[:,:640] for sc in scenes]
            pairs=np.stack([cv2.cvtColor(cv2.imread(str(root/h['images']['model'])),cv2.COLOR_BGR2RGB) for h in history])
            pix=torch.from_numpy(pairs).permute(0,3,1,2).float()/255
            obs={'overhead':pix[None,:,:,:,:128].contiguous(),'wrist':pix[None,:,:,:,128:].contiguous(),
                 'agent_pos':torch.tensor([[k['q'] for k in history]],dtype=torch.float32)}
            anchor=torch.tensor([r['plan']['anchor']],dtype=torch.float32)
            with torch.inference_mode():
                base=model.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
            joints,p=adapter.predict(raw,np.array(r['q']),base,calibrate=False)
            revised=adapter.plan(p,r['q'],base,True)
            history_centers.append(p.copy())
            p_tr=np.median(history_centers[-5:],axis=0)
            tracked=adapter.plan(p_tr,r['q'],base,True)
            combined=adapter.plan(p,r['q'],base,True,scenes[-1][:,640:],np.array([h['q'] for h in history]))
            combined_tr=adapter.plan(p_tr,r['q'],base,True,scenes[-1][:,640:],np.array([h['q'] for h in history]))
            goal=np.r_[ref,1.]@adapter.par['matrix']
            start=adapter.fk(r['q']);d0=np.linalg.norm(goal-start)
            table.append(dict(seconds=(r['stamp']-anchors[0]['stamp'])/1e9,
                original_advance_mm=d0-np.linalg.norm(adapter.fk(base[-1])-goal),
                new_advance_mm=d0-np.linalg.norm(adapter.fk(joints[-1])-goal),initial_mm=d0,
                wood_error_px=np.linalg.norm(p-ref),aperture_mm=r['q'][6],
                registered_advance_mm=d0-np.linalg.norm(adapter.fk(revised[-1])-goal),
                tracked_registered_advance_mm=d0-np.linalg.norm(adapter.fk(tracked[-1])-goal),
                fused_advance_mm=d0-np.linalg.norm(adapter.fk(combined[-1])-goal),
                tracked_fused_advance_mm=d0-np.linalg.norm(adapter.fk(combined_tr[-1])-goal),
                tracked_wood_error_px=np.linalg.norm(p_tr-ref)))
        stats[name]=table
        print('archived field plans',name,len(table),flush=True)
    audit=json.loads((out/'audit_inputs.json').read_bytes()) if (out/'audit_inputs.json').exists() else {}
    audit['field']=stats
    rel.dump(out/'audit_inputs.json',audit)


def object_only(work,out):
    """Controlled image intervention using two real training cutouts on a fixed held-out tabletop."""
    import torch,cv2
    from zipfile import ZipFile
    torch.set_num_threads(2);cv2.setNumThreads(1)
    xy,rel,_=setup(work)
    _,_,records,geo,goals,fk,d,training,starts,target,_=xy.load_windows(work)
    from annotate import images
    adapter=B1OfflineAlignment(work,out)
    _,coord=rel.paths(work)
    index={r['key']:r for r in json.loads((coord/'frames.json').read_bytes())}
    keys={r['episode']:r['key'] for r in index.values() if r['phase']=='release'}
    polygons={32:[[23,18],[54,18],[57,24],[56,59],[21,59],[20,54]],
              52:[[26,16],[57,18],[59,24],[56,54],[29,51],[22,46]]}
    pos={ep:i for i,ep in enumerate(d['episodes']) if ep not in training}
    def read(z,key):
        return cv2.imdecode(np.frombuffer(z.read(key+'.jpg'),np.uint8),1)
    result=[]
    with ZipFile(coord/'frames.zip') as z:
        cuts={}
        for key,points in polygons.items():
            k=f'{key:02d}_review';r=index[k]
            assert r['train']
            u,v=np.rint(r['uv']).astype(int)
            patch=read(z,k)[v-32:v+48,u-40:u+40].copy()
            mask=np.zeros((80,80),np.uint8)
            cv2.fillConvexPoly(mask,np.array(points,dtype=np.int32),255)
            cuts[key]=patch,cv2.GaussianBlur(mask,(3,3),.6)[:,:,None]/255.
        for a,b in geo['pairs']:
            for ep,donor in ((a,b),(b,a)):
                i=pos[ep];r=records[ep];t=d['indices'][i]
                wrist=cv2.imread(str(images(r['folder'],'wrist')[r['first']+t]))
                backdrop=read(z,keys[ep])
                qhist=r['q'][t-7:t+1]
                axis=goals[donor]-goals[ep];axis/=np.linalg.norm(axis)
                for key,(patch,mask) in cuts.items():
                    loc=[];ends=[]
                    for name in (ep,donor):
                        im=backdrop.copy()
                        u,v=np.rint(records[name]['uv']).astype(int)
                        box=im[v-32:v+48,u-40:u+40]
                        box[:]=np.rint((1-mask)*box+mask*patch).astype(np.uint8)
                        # Identical frame repeated is the static-camera limiting case.
                        commands,p=adapter.predict([im]*8,qhist[-1],d['base'][i],qhistory=qhist,wrist=wrist)
                        loc.append(np.linalg.norm(p-[u,v]));ends.append(adapter.fk(commands[-1]))
                    result.append(dict(recipient=ep,donor=donor,training_cutout=key,signed_mm=(ends[1]-ends[0])@axis,
                         original_pixel_error=loc[0],new_pixel_error=loc[1]))
    p=out/'audit_inputs.json';audit=json.loads(p.read_bytes())
    audit['object_only']=result;rel.dump(p,audit)
    print('controlled block-only changes',len(result),flush=True)


def second(work,out,stage):
    import shutil
    rev=out/'causal_appearance'
    rev.mkdir(exist_ok=True)
    if stage=='train':
        for f in ['fit.json','protocol.md','private_policy.npz']:shutil.copyfile(out/f,rev/f)
        train_heatmap(work,rev,augmented=True)
        fit_fusion(work,rev)
    elif stage=='settle':
        old_eval=rev/'outcomes.json'
        parent=out/'audit_inputs.json'
        if parent.exists() and old_eval.exists():
            _,rel,_=setup(work)
            previous=json.loads((rev/'inspection.json').read_bytes()).get('settle',{})
            archive=json.loads(parent.read_bytes())
            if previous and 'height_gate' not in previous and 'phase_study' in archive:
                archive['before_height_gate']=dict(fit=previous,phase_study=archive['phase_study'])
                rel.dump(parent,archive)
        settle_fit(work,rev)
    elif stage=='eval':
        evaluate(work,rev,heat=True)
        field(work,rev)
        object_only(work,rev)
        parent=json.loads((out/'audit_inputs.json').read_bytes())
        parent['appearance_training']={f:json.loads((rev/(f+'.json')).read_bytes())
            for f in ('inspection','outcomes','audit_inputs')}
        _,rel,_=setup(work);rel.dump(out/'audit_inputs.json',parent)


def progression(work,out):
    """Evaluate the final policy at several real pre-contact times; outcomes are grouped by episode."""
    import cv2,torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    xy,rel,_=setup(work)
    from annotate import images
    from fit_gripper import flange
    records,geo,goals,fk=rel.data(work)
    directory=out/'causal_appearance'
    adapter=B1OfflineAlignment(work,directory)
    groups=[]
    for r in records:
        ep=r['episode']
        if ep not in goals:continue
        stop=min(v for v in (r['close'],r['contact']) if v is not None)
        files=images(r['folder'],'overhead')[r['first']:]
        wf=images(r['folder'],'wrist')[r['first']:]
        contact=fk(r['q'][r['contact']])
        for off in (36,30,24,18,12,8,4):
            t=stop-off
            if t<7:continue
            qhistory=r['q'][t-7:t+1]
            pix=[cv2.imread(str(f)) for f in files[t-7:t+1]]
            wrist=cv2.imread(str(wf[t]))
            uv,_=adapter.det.estimate(pix)
            if adapter.steady is None:goal=fused(uv,qhistory,wrist,adapter.par,adapter.fusion,flange)
            else:goal=settle_goal(uv,qhistory,wrist,adapter.par,adapter.steady,flange)
            s=fk(qhistory[-1])
            end=move(adapter.par,s,goal,'feedback','fused')
            ref=fk(r['q'][min(t+8,r['contact'])])
            groups.append(dict(episode=ep,offset=off,original_gap_mm=np.linalg.norm(contact-s),
                predicted_gap_mm=np.linalg.norm(contact-end),goal_contact_mm=np.linalg.norm(contact-goal),
                hold_error_mm=np.linalg.norm(s-ref),prediction_error_mm=np.linalg.norm(end-ref),
                projected_reference_gain_mm=np.linalg.norm(s-goals[ep])-np.linalg.norm(end-goals[ep])))
    p=out/'audit_inputs.json';audit=json.loads(p.read_bytes())
    audit['phase_study']=groups;rel.dump(p,audit)
    print('chronological archive checks',len(groups),flush=True)


def independent_visibility(work,out):
    """Fixed visual references for two visible cubes in seven previously excluded recordings."""
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    _,rel,_=setup(work)
    # Manually lock before inference. Five remaining clips hide the top-face centre at frame 15.
    references={'222417':(234,343),'222426':(237,349)}
    root=work/'batches/B1_20261123/failed'
    adapter=B1OfflineAlignment(work,out/'causal_appearance')
    rows=[]
    for ep in sorted(root.iterdir()):
        files=sorted((ep/'episode0/camera/color/overhead').glob('*.jpg'),key=lambda p:float(p.stem))
        if len(files)<16:continue
        label=references.get(ep.name.split('_')[1])
        row=dict(episode=ep.name,frame_index=15,reference_uv=label)
        if label is not None:
            uv,_=adapter.det.estimate([cv2.imread(str(p)) for p in files[8:16]])
            row.update(estimate_uv=uv.tolist(),error_px=float(np.linalg.norm(uv-label)))
        rows.append(row)
    assert len(rows)==7 and sum('error_px' in r for r in rows)==2,rows
    dest=out/'audit_inputs.json';data=json.loads(dest.read_bytes())
    data['independent_visibility']=rows
    rel.dump(dest,data)
    print('independent visibility',rows,flush=True)


def replay_final(work,out):
    """Replay original checkpoint and final adapter from archived camera files, independent of cached actions."""
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    xy,rel,_=setup(work)
    from annotate import images
    from probe import load_model
    from train_contrast import image_rgb
    records,geo,goals,fk=rel.data(work)
    model,checkpoint=load_model(work,'baseline')
    rev=out/'causal_appearance'
    adapter=B1OfflineAlignment(work,rev)
    assert adapter.steady is not None
    mode='settle/feedback'
    results=json.loads((rev/'outcomes.json').read_bytes())
    natural={r['episode']:r for r in results['natural'] if r['mode']==mode}
    changed={(r['recipient'],r['donor']):r for r in results['directions'] if r['mode']==mode}
    frames={}
    for r in records:
        ep=r['episode']
        if ep not in geo['test_episodes']:continue
        t=(r['close'] if r['close'] is not None else r['contact'])-18
        raw={}
        for view in ('overhead','wrist'):
            files=images(r['folder'],view)[r['first']:]
            raw[view]=[cv2.imread(str(f)) for f in files[t-7:t+1]]
        obs={k:torch.from_numpy(np.stack([image_rgb(im,128) for im in v])[None]).float()/255
             for k,v in raw.items()}
        q=r['q'][t-7:t+1]
        obs['agent_pos']=torch.from_numpy(q[None])
        frames[ep]=(raw,obs,q,torch.from_numpy(r['cmd'][None,t-1]),fk(r['q'][t+8]),fk(r['q'][r['contact']]))
    ends={}
    errors=[]
    height_errors=[]
    aperture_errors=[]
    distances=[]
    with torch.inference_mode():
        trials=[(e,e) for e in natural]+list(changed)
        for ep,donor in trials:
            raw,obs,q,anchor,measured,_=frames[ep]
            overhead,source_obs,*_=frames[donor]
            observation=dict(obs,overhead=source_obs['overhead'])
            base=model.predict_action(observation,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
            joint,_=adapter.predict(overhead['overhead'],q[-1],base,qhistory=q,wrist=raw['wrist'][-1])
            passthrough,_=adapter.predict([],q[-1],base,approach=False)
            np.testing.assert_array_equal(passthrough,base)
            base_xyz,_=xy.pose(base[:,:6].astype(float),adapter.par['tip'])
            xyz,_=xy.pose(joint[:,:6],adapter.par['tip'])
            height_errors.append(np.max(np.abs(xyz[:,2]-base_xyz[:,2])))
            aperture_errors.append(np.max(np.abs(joint[:,6]-base[:,6])))
            p=fk(joint[-1])
            initial=fk(q[-1])
            ref=frames[donor][-1]
            distances.append(dict(recipient=ep,donor=donor,initial_to_contact_mm=np.linalg.norm(initial-ref),
                final_to_contact_mm=np.linalg.norm(p-ref),base_to_contact_mm=np.linalg.norm(fk(base[-1])-ref),
                initial_to_block_mm=np.linalg.norm(initial-goals[donor]),final_to_block_mm=np.linalg.norm(p-goals[donor]),
                base_to_block_mm=np.linalg.norm(fk(base[-1])-goals[donor]),
                same_pose_eighth_step_error_mm=np.linalg.norm(initial-measured) if ep==donor else None))
            if ep==donor:
                ends[ep]=p
                errors.append(abs(np.linalg.norm(p-measured)-natural[ep]['error_mm']))
            else:
                delta=goals[donor]-goals[ep]
                signed=(p-ends[ep])@delta/np.linalg.norm(delta)
                errors.append(abs(signed-changed[ep,donor]['signed_mm']))
                advance=np.linalg.norm(fk(q[-1])-goals[donor])-np.linalg.norm(p-goals[donor])
                errors.append(abs(advance-changed[ep,donor]['toward_new_mm']))
    assert len(natural)==33 and len(changed)==32 and max(errors)<1e-4
    assert max(height_errors)<1e-4 and max(aperture_errors)==0
    try:
        adapter.predict(overhead['overhead'],q[-1],base)
    except ValueError as ex:
        assert 'wrist image' in str(ex)
    else:
        raise AssertionError('Incomplete final-candidate input accepted')
    p=out/'audit_inputs.json'
    data=json.loads(p.read_bytes())
    data['raw_replay']=dict(mode=mode,original_checkpoint_sha256=checkpoint,
        private_hashes={name:rel.digest(rev/name) for name in ('private_policy.npz','private_heatmap.pt','private_settle.npz')},
        natural_count=len(natural),swapped_count=len(changed),scalar_count=len(errors),
        max_scalar_difference_mm=max(errors),max_height_difference_mm=max(height_errors),
        max_gripper_difference=max(aperture_errors),exact_bypass_count=len(trials),missing_feedback_or_wrist_rejected=True)
    print('independent camera-to-plan replay',data['raw_replay'],flush=True)
    data['raw_replay']['distances']=distances
    rel.dump(p,data)


if __name__=='__main__':
    work,out=map(Path,sys.argv[1:3])
    if sys.argv[3]=='inspect':inspect(work,out)
    elif sys.argv[3]=='fit':fit(work,out)
    elif sys.argv[3]=='train_heatmap':train_heatmap(work,out)
    elif sys.argv[3]=='registration':fit_registration(work,out)
    elif sys.argv[3]=='fusion':fit_fusion(work,out)
    elif sys.argv[3] in ('eval','eval_heat'):evaluate(work,out,sys.argv[3]=='eval_heat')
    elif sys.argv[3]=='field':field(work,out)
    elif sys.argv[3]=='object_only':object_only(work,out)
    elif sys.argv[3].startswith('v2_'):second(work,out,sys.argv[3][3:])
    elif sys.argv[3]=='progression':progression(work,out)
    elif sys.argv[3]=='replay_final':replay_final(work,out)
    elif sys.argv[3]=='independent_visibility':independent_visibility(work,out)
