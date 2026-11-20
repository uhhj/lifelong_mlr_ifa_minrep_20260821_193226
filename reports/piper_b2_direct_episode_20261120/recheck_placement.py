"""Recheck marginal releases against empty-dish images from separate recordings."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from check_b2_scene import overhead, background, dish_outline, block


def recheck(base,name):
    cv2.setNumThreads(1)
    root=base/"recordings"
    ids=("20261120_165700_4177037","20261120_170331_4179474","20261120_170355_4179785")
    refs=[background(overhead(root/ref)) for ref in ids if ref!=name]
    ref=np.median(refs,axis=0).astype("uint8")
    hull=dish_outline(ref)
    if hull is None:
        raise ValueError("Independent reference rim unavailable")
    x,y,w,h=cv2.boundingRect(hull)
    mask=np.zeros(ref.shape[:2],"uint8")
    cv2.fillConvexPoly(mask,hull,1)
    roi=np.s_[y-24:y+h+24,x-24:x+w+24]
    mask=cv2.erode(mask,np.ones((5,5),"uint8"))[roi].astype(bool)
    ref_v=cv2.cvtColor(ref[roi],cv2.COLOR_BGR2HSV)[:,:,2]
    ep=root/name
    paths=overhead(ep)
    hits=[block(cv2.imread(str(f))[roi],ref_v,mask) for f in paths]
    rows=[json.loads(line) for line in (ep/"alignment.jsonl").read_text().splitlines()]
    assert len(hits)==len(rows)
    start=next(i for i,r in enumerate(rows) if r["target"] is not None)
    width=np.array([r["feedback"]["gripper_mm"] for r in rows])
    width=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(width,1,mode="edge"),3),axis=1)
    end_candidates=[h for h in hits[-12:] if h]
    def close(a,b,radius):
        return a and b and np.hypot(a[0]-b[0],a[1]-b[1])<=radius
    end=max(end_candidates,key=lambda a:sum(bool(close(a,h,9)) for h in end_candidates),default=None)
    same=np.array([bool(close(h,end,10)) for h in hits])
    clear=np.array([bool(h and same[i] and h[2]>=5) for i,h in enumerate(hits)])
    # Count support across seven acquired frames. Modest lighting/outline variation
    # is tolerated while the block stays in its independently estimated final location.
    initial=sum(bool(h) for h in hits[:30])
    terminal=bool(same[-2:].any() and same[-12:].sum()>=7)
    placement=None
    for i in range(max(6,start),len(hits)):
        clear_count=int(clear[i-6:i+1].sum())
        if clear_count<5:continue
        # Require an observed opening transition (three-frame median) near the
        # dish, preceded by a closing transition, within 1.6 s of a clear window.
        for j in range(max(start+2,i-48),i+1):
            pre=max(start,j-15)
            low=pre+int(np.argmin(width[pre:j+1]))
            opened=width[j]-width[low]
            closed=width[start:low+1].max()-width[low]
            if opened<5 or closed<5: continue
            near=hits[max(start,j-10):min(i+1,j+11)]
            if not any(close(k,end,26) for k in near):continue
            placement=dict(open_at_row=j,clear_window_end=i,clear_in_seven=clear_count,
                           opening_mm=round(float(opened),1),closing_mm=round(float(closed),1))
            break
        if placement:break
    return dict(start_hits=initial,last12_same=int(same[-12:].sum()),terminal_supported=terminal,
                clear_near_final_total=int(clear.sum()),max_clear_in_seven=int(np.convolve(clear.astype(int),np.ones(7,dtype=int),mode="valid").max()),
                release=placement,confirmed=bool(initial<15 and terminal and placement))


if __name__=="__main__":
    print(json.dumps(recheck(Path(sys.argv[1]),sys.argv[2])))
