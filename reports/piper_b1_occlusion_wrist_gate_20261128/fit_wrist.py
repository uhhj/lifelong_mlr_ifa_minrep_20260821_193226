"""Fit image-to-gripper offsets on the training episodes, cross-checking by whole episode."""
from basis import *
from wrist_xy import features
from scipy.optimize import least_squares
def fit(X,Y):
    initial=np.linalg.lstsq(X,Y,rcond=None)[0]
    return least_squares(lambda b:np.r_[((X@b.reshape(-1,2)-Y)).ravel(),.6*b[2:]],initial.ravel(),loss='huber',f_scale=3).x.reshape(-1,2)
def run():
    m=setup();cal=m.relative.calibration
    from assess import reviewed
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    newer=[r for r in reviewed(work,prior) if r['train'] and r['close'] is not None and r['contact'] is not None]
    groups=[list(rows('train')),[(r['episode'],dict(**rp.load_record(r['folder'],r['first']),close=r['close'],contact=r['contact'])) for r in newer]]
    matrices=[];meta=[]
    for g,group in enumerate(groups):
        xs=[];ys=[];eps=[]
        for ep,r in group:
            c=r['close'];seq=r['rows'];contact=r.get('contact')
            if contact is None:contact=next((i for i in range(max(0,c-6),len(seq)-2) if all(rp.valid(s) and rp.at(s,'feedback')[6]<=49 for s in seq[i:i+3])),None)
            if contact is None:continue
            goal=m.aligner.fk(rp.at(seq[contact],'feedback'))
            for i in sorted(set(max(0,c-k) for k in (48,40,32,26,22,18,15,12,9,6,3,0))):
                if not rp.valid(seq[i]):continue
                q=rp.at(seq[i],'feedback');tf=m.aligner.flange(np.rad2deg(q[:6]));tip=tf[:3,3]+tf[:3,:3]@m.aligner.par['tip']
                f=features(rp.imaging(r,i,'wrist'),tip,tf[:3,:3],cal)
                if f is None or np.linalg.norm(goal-tip[:2])>62:continue
                x,b=f;xs.append(x);ys.append(np.linalg.solve(b,goal-tip[:2]));eps.append(ep)
        X=np.array(xs);Y=np.array(ys);E=np.array(eps)
        assert len(set(eps))>=20
        errors=[]
        for ep in sorted(set(eps)):
            test=E==ep;M=fit(X[~test],Y[~test]);errors.extend(np.linalg.norm(X[test]@M-Y[test],axis=1))
        low,high=np.percentile(X,[1,99],axis=0);margin=np.maximum((high-low)*.16,[0,3,3,3,1,2,2])
        matrices.append(dict(coef=fit(X,Y).tolist(),lo=(low-margin).tolist(),hi=(high+margin).tolist(),uncertainty=float(np.median(errors))))
        meta.append(dict(episodes=len(set(eps)),observations=len(xs),loo_median_mm=float(np.median(errors)),loo_p90_mm=float(np.percentile(errors,90))))
    dump('private_wrist_fit.json',matrices);dump('fit_meta.json',meta);print(meta,flush=True)
if __name__=='__main__':run()
