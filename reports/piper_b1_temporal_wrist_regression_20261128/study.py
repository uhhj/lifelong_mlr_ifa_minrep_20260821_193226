"""Separate heatmap location error, scene motion, and calibration error using archived observations."""
from survey import *
from scipy.optimize import least_squares
import hashlib

def purple(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(92,35,24),(153,255,240))
    mask[:265]=0;mask[:,:350]=0
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in sorted(contours,key=cv2.contourArea,reverse=True):
        x,y,w,h=cv2.boundingRect(c)
        if not 30<w<130 or not 38<h<150 or cv2.contourArea(c)<.55*w*h:continue
        m=cv2.moments(c)
        return np.array([m['m10']/m['m00'],m['m01']/m['m00']])
    return None

def fit(uv,xy):
    A=np.c_[uv,np.ones(len(uv))]
    init=np.linalg.lstsq(A,xy,rcond=None)[0]
    return least_squares(lambda v:(A@v.reshape(3,2)-xy).ravel(),init.ravel(),loss='huber',f_scale=4).x.reshape(3,2)

def study():
    cv2.setNumThreads(1)
    rp.torch.set_num_threads(2)
    model=rp.models()[1];det=model.aligner.det;flange=model.aligner.flange;tip=model.aligner.par['tip']
    gt=np.asarray(read(work/'offline_diagnostics/b1_vision_contrasts_20261124/final.json')['geometry']['table_map'])
    points=read(root/'points.json')
    samples={}; sheets=[];refs=[]
    for group in ('train','test'):
        fitrows=[];tiles=[]
        for i,(ep,r) in enumerate(rows(group)):
            c=r['close']; ticks=[0,max(0,(c-18)//2),max(0,c-18)]
            label=points[group][i]
            if label is None:continue
            v=np.array(label[:2])/.75+[96,214];j=ticks[label[2] if len(label)>2 else 0]
            source=rp.imaging(r,j,'overhead');late=rp.imaging(r,ticks[-1],'overhead')
            ref=purple(source);end_ref=purple(late)
            q=np.array([r0['feedback']['joints_rad'] if r0.get('feedback') else [0]*6 for r0 in r['rows']])
            ff=flange(np.rad2deg(q)); xyz=ff[:,:3,3]+ff[:,:3,:3]@tip
            contact=next(k for k in range(max(0,c-6),len(q)-2) if all(rp.valid(x) and rp.at(x,'feedback')[6]<=49 for x in r['rows'][k:k+3]))
            near=range(max(0,c-8),min(c+27,len(q)))
            low=min(near,key=lambda k:xyz[k,2])
            rec=dict(ep=ep,index=i,label=v.tolist(),ref=None if ref is None else ref.tolist(),
                     late_ref=None if end_ref is None else end_ref.tolist(),goal=xyz[contact,:2].tolist(),
                     contact_z=float(xyz[contact,2]),low_goal=xyz[low,:2].tolist(),low_z=float(xyz[low,2]),contact_i=contact,close=c)
            if group=='test':
                early=[rp.imaging(r,t,'overhead') for t in range(8)]
                later=[rp.imaging(r,t,'overhead') for t in range(c-25,c-17)]
                epix,_=det.estimate(early);lpix,_=det.estimate(later)
                rec.update(early=epix.tolist(),late=lpix.tolist())
                ims=[rp.imaging(r,k,'wrist') for k in (c-18,c,contact,min(c+18,len(q)-1))]
                tile=np.hstack([cv2.resize(im,(176,132)) for im in ims]);cv2.putText(tile,str(i),(3,14),0,.48,(0,240,255),1)
                tiles.append(tile)
            if i==0 and group=='train':refs.append(source)
            fitrows.append(rec)
        samples[group]=fitrows
        if tiles:
            for i in range(0,len(tiles),10):cv2.imwrite(str(root/f'wrist{i}.jpg'),np.vstack(tiles[i:i+10]))
    train=samples['train']
    usable=[x for x in train if x['ref'] is not None]
    pivot=np.median([x['ref'] for x in usable],axis=0)
    def uv(x):return np.array(x['label'])-x['ref']+pivot
    mat=fit([uv(x) for x in usable],np.array([x['goal'] for x in usable]))
    rawmat=fit([x['label'] for x in train],np.array([x['goal'] for x in train]))
    recs=rp.read(work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json')
    new=work/'batches/B1_20261123'/recs['ranking'][0]['source']
    example=rp.load_record(new);refs.append(rp.imaging(example,0,'overhead'))
    cv2.imwrite(str(root/'compare.jpg'),np.hstack(refs))
    for x in samples['test']:
        hand=np.array(x['label']);goal=np.array(x['goal'])
        delta=np.array(x['late_ref'])-x['ref'] if x['ref'] is not None and x['late_ref'] is not None else np.zeros(2)
        moved=hand+delta
        x['errors']=dict(early_px=float(np.linalg.norm(x['early']-hand)),late_px=float(np.linalg.norm(x['late']-moved)),
            heatmap_mm=float(np.linalg.norm(np.r_[x['late'],1]@gt-goal)),manual_current_map_mm=float(np.linalg.norm(np.r_[moved,1]@gt-goal)),
            old_map_mm=float(np.linalg.norm(np.r_[hand,1]@rawmat-goal)),
            relative_old_map_mm=float(np.linalg.norm(np.r_[uv(x),1]@mat-goal)) if x['ref'] else None,marker_motion_px=float(np.linalg.norm(delta)))
    np.savez(root/'private_scene_map.npz',matrix=mat,raw_matrix=rawmat,pivot=pivot)
    (root/'diag.json').write_text(json.dumps(samples))
    print('old calibration labels',len(train),'anchor',len(usable),'medians',[(key,round(float(np.median([v['errors'][key] for v in samples['test'] if v['errors'][key] is not None])),2)) for key in samples['test'][0]['errors']],flush=True)

if __name__=='__main__':study()
