"""Instrument the exact frozen gate. Missing edge measurements leave width/position unscored."""
from basis import *
from collections import Counter
from unittest.mock import patch
def diagnose():
    m=setup()
    import temporal
    from b1_placement import B1Placement
    from b1_alignment_gate import wrist_centered
    chosen={e for e,r in read(edge/'misses.json').items() if r['group']=='archive' and r['reason']=='no_wrist_confirmation'}
    assert len(chosen)==15
    evidence={};boards=[];all_frames={}
    original_observe=temporal.TemporalTarget.observe
    original_center=temporal.wrist_centered
    original_gate=B1Placement.next_goal
    last={}
    def center(box,opening):
        last['supported']=list(map(float,box));last['opening']=opening
        return original_center(box,opening)
    def observe(obj,stamp,frame,opening,box):
        last.clear();last['history']=len(obj.faces);last['seed']=box is not None;last['opening']=opening
        return original_observe(obj,stamp,frame,opening,box)
    for ep,rec in rows('test'):
        if ep not in chosen:continue
        c=rec['close']
        stamps={r['sample_monotonic_ns']:i for i,r in enumerate(rec['rows'])}
        frames=[]
        def gate(obj,stamp,intent,xyz,width,*,vision):
            i=stamps[stamp];box=last.get('supported');size=xok=yok=None
            if box is not None:
                x,y,w,h=box;clear=max(0.,(width-35)/2-2)*w/35
                xok=bool(clear>0 and abs(x+w/2-355)<=min(42.,clear))
                yok=bool(276<=y<=372 and y+h>=405)
                size=bool(vision['size_ok'])
            flags=dict(i=i-c,intent=bool(intent),coarse=bool(vision.get('view') is not None and vision['distance_mm']<=20),
                height=bool(vision['depth_ok']),edge=box is not None,size=size,jaw_x=xok,jaw_depth=yok,
                metric=bool(vision.get('fresh_metric_center') and wrist_centered(vision['wrist_box'],width)),
                aligned=bool(vision['aligned']),seed=last['seed'],history=last['history'],width_px=None if box is None else round(box[2],1),
                z=round(float(xyz[2]),2),gap_mm=round(vision['distance_mm'],1))
            frames.append(flags)
            return original_gate(obj,stamp,intent,xyz,width,vision=vision)
        with patch.object(temporal.TemporalTarget,'observe',observe),patch.object(temporal,'wrist_centered',center),patch.object(B1Placement,'next_goal',gate):
            assert rp.close_event(m,rec) is None
        tick=next(f for f in frames if f['i']==0)
        all_frames[ep]=frames
        near=[f for f in frames if -8<=f['i']<=8]
        evidence[ep]=dict(window=near,expert=tick,counts={k:sum(f[k] is True for f in near) for k in ('intent','coarse','height','edge','size','jaw_x','jaw_depth','metric','aligned')})
        ims=[]
        for j in (-6,0,5):
            im=rp.imaging(rec,c+j,'wrist');ims.append(cv2.resize(im,(224,168)))
        band=np.hstack(ims);cv2.putText(band,ep[9:15],(2,15),0,.46,(255,255,255),1)
        boards.append(band)
    for i in range(0,len(boards),5):cv2.imwrite(str(dest/f'case{i}.jpg'),np.vstack(boards[i:i+5]))
    dump('diagnosis.json',evidence);dump('private_frames.json',all_frames)
    print('exact expert-frame patterns',Counter(tuple(k for k in ('intent','coarse','height','edge','size','jaw_x','jaw_depth') if q['expert'][k] is False) for q in evidence.values()),flush=True)
if __name__=='__main__':diagnose()
