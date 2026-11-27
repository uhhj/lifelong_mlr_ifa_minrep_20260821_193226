"""Calibrate viewing directions using contact positions and visible expert cube faces."""
from itertools import permutations,product
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from wrist_target import top_face


def fit(work):
    from fit_gripper import flange
    root=Path(__file__).resolve().parent
    labels=json.loads((root/'refs.json').read_bytes())
    with np.load(root/'private_refs.npz') as d:
        imgs,xyz,qs=d['im'],d['xy'],d['q']
    data=[]
    for i,(lab,views) in enumerate(zip(labels,imgs)):
        seen=set()
        for j,(t,pair) in enumerate(zip(lab['indices'][:6],views[:6])):
            if t in seen or t>=min(lab['close'],lab['contact']):continue
            seen.add(t)
            face=top_face(pair[1])
            if face is None or not 55<=face[2]<=145:continue
            R=flange(np.rad2deg(qs[i,j]))[:3,:3]
            data.append((i,lab['train'],xyz[i,j],R,xyz[i,-1,:2],face))
    ids,train,P,R,C,F=map(np.array,zip(*data))
    def solve(mask):
        p,rot,contact,faces=P[mask],R[mask],C[mask],F[mask]
        w=np.sqrt(np.array([1/sum(ids[mask]==i) for i in ids[mask]]))
        def residual(v):
            cam=Rotation.from_rotvec(v[:3]).as_matrix()
            f=v[6];z=v[7]
            # Reconstruct world coordinates by intersecting the calibrated ray
            # with the plane at the top of the block.
            A=rot@cam;origin=p+rot@v[3:6]
            ray=(A@np.c_[(faces[:,:2]-[320,240])/f,np.ones(len(p))][:,:,None])[:,:,0]
            zs=(z-origin[:,2])/np.where(abs(ray[:,2])>.05,ray[:,2],np.where(ray[:,2]>=0,.05,-.05))
            xy=origin[:,:2]+ray[:,:2]*zs[:,None]
            # The observed face width regularizes depth and focal length.
            extent=f*35*np.sqrt(np.clip(abs(ray[:,2]),.2,1.2))/np.clip(zs,30,700)
            return np.r_[((xy-contact)*w[:,None]).ravel(),.12*(extent-faces[:,2])*w,
                         2*np.minimum(zs-40,0)*w,.01*(f-520),.035*(z-18),.009*v[3:6]]
        solutions=[]
        for perm in permutations(range(3)):
            for signs in product((-1,1),repeat=3):
                Q=np.eye(3)[:,perm]*signs
                if np.linalg.det(Q)<0 or np.median((rot@Q)[:,2,2])>-.55:continue
                v0=np.r_[Rotation.from_matrix(Q).as_rotvec(),np.zeros(3),520.,18.]
                measured=np.c_[(faces[:,:2]-[320,240])*35/faces[:,2,None],520*35/faces[:,2]]
                rel=(rot.transpose(0,2,1)@(np.c_[contact,np.full(len(p),18)]-p)[:,:,None])[:,:,0]
                v0[3:6]=np.clip(np.median(rel-measured@Q.T,axis=0),-185,185)
                res=least_squares(residual,v0,bounds=([-8]*3+[-200]*3+[300,-15],[8]*3+[200]*3+[850,70]),
                                  loss='soft_l1',f_scale=3,max_nfev=120)
                solutions.append((res.cost,res.x))
        assert solutions
        v=min(solutions,key=lambda s:s[0])[1]
        return dict(R=Rotation.from_rotvec(v[:3]).as_matrix().tolist(),offset=v[3:6].tolist(),f=float(v[6]),z=float(v[7]))
    def predict(c,mask):
        A=R[mask]@c['R'];origin=P[mask]+R[mask]@np.asarray(c['offset'])
        ray=(A@np.c_[(F[mask,:2]-[320,240])/c['f'],np.ones(mask.sum())][:,:,None])[:,:,0]
        xy=origin[:,:2]+ray[:,:2]*((c['z']-origin[:,2])/ray[:,2])[:,None]
        err=np.linalg.norm(xy-C[mask],axis=1)
        return dict(n=len(err),episodes=len(set(ids[mask])),median_mm=float(np.median(err)),p90_mm=float(np.percentile(err,90)))
    out={}
    for key,mask in [('train',train),('independent',~train)]:
        c=solve(mask)
        out[key]=dict(calibration=c,own=predict(c,mask),opposite=predict(c,~mask))
    (root/'camera_models.json').write_text(json.dumps(out,indent=2)+'\n')
    print('geometric validation',[(k,r['opposite']) for k,r in out.items()],flush=True)
