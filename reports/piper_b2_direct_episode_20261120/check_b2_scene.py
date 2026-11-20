"""Check placement in the right pale dish using only saved B2 images and feedback."""
import json
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np


def overhead(episode):
    folder = episode/"episode0/camera/color/overhead"
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")),key=lambda p:Decimal(p.stem))


def background(paths):
    indices = [min(i,len(paths)-1) for i in (2,6,10,15,20)]
    return np.median([cv2.imread(str(paths[i])) for i in indices],axis=0).astype("uint8")


def dish_outline(image):
    gray = cv2.cvtColor(cv2.GaussianBlur(image,(5,5),0),cv2.COLOR_BGR2GRAY)
    candidates = []
    for low,high in ((15,40),(10,30),(25,75)):
        edges = cv2.morphologyEx(cv2.Canny(gray,low,high),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if abs(x-361)>14 or abs(y-189)>13 or abs(w-171)>23 or abs(h-110)>19:
                continue
            area=cv2.contourArea(contour)
            if not .72*w*h<area<.99*w*h:
                continue
            score=abs(x-361)+abs(y-189)+.5*(abs(w-171)+abs(h-110))
            candidates.append((score,cv2.convexHull(contour)))
    return min(candidates,key=lambda c:c[0])[1] if candidates else None


def block(image, ref_value, mask):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue,sat,val=cv2.split(hsv)
    # Relative brightness compensates per frame for exposure/white-balance changes.
    diff=ref_value.astype("int16")-val.astype("int16")
    baseline=np.median(diff[mask][::5])
    warmth=sat.astype("int16")-np.median(sat[mask][::5])
    darker=diff-baseline
    color=(hue>=3)&(hue<=44)&(sat>=20)&(sat<176)&(val>43)&mask
    evidence=(darker>19, darker>10, (darker>15)|((warmth>15)&(darker>-8)), (warmth>18)&(darker>-12))
    hits=[]
    for variant,contrast in enumerate(evidence):
        binary=cv2.morphologyEx((color&contrast).astype("uint8"),cv2.MORPH_OPEN,np.ones((3,3),"uint8"))
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),"uint8"))
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area=cv2.contourArea(c)
            (x,y),(w,h),_=cv2.minAreaRect(c)
            if 240<=area<=2400 and min(w,h)>=14 and max(w,h)<=66 and min(w,h)>=.42*max(w,h) and area>=.48*w*h:
                hits.append((area,x,y,cv2.boundingRect(c),variant))
    if not hits:
        return None
    def score(candidate):
        area,x,y,_,_=candidate
        support={other[4] for other in hits if np.hypot(other[1]-x,other[2]-y)<=11}
        return len(support),-abs(np.log(area/1050))
    area,x,y,box,_=max(hits,key=score)
    dark=(val<58).astype("uint8")
    _,labels,stats,_=cv2.connectedComponentsWithStats(dark)
    bx,by,bw,bh=box
    ids=[i for i,(l,t,w,h,a) in enumerate(stats) if i and a>=80 and
         not (l>=bx-2 and t>=by-2 and l+w<=bx+bw+2 and t+h<=by+bh+2 and a<.6*area)]
    if len(ids):
        distance=cv2.distanceTransform((~np.isin(labels,ids)).astype("uint8"),cv2.DIST_L2,5)
        clearance=float(distance[int(round(y)),int(round(x))])-np.sqrt(area/np.pi)
    else:
        clearance=100.
    return (x,y,clearance)


def summarize(hits, widths, times, valid):
    n=len(hits)
    result=dict(first30=sum(h is not None for h in hits[:30]),last30=sum(h is not None for h in hits[-30:]),
                terminal=hits[-1] is not None, terminal_three=[hits[i] is not None for i in (-11,-6,-1)])
    # Require the same position near the end. Exposure can hide a single outline.
    recent=[h for h in hits[-12:] if h is not None]
    def support(h):
        return sum(np.hypot(k[0]-h[0],k[1]-h[1])<=8 for k in recent)
    end=max(recent,key=support,default=None)
    if end is not None and (support(end)<5 or not any(h and np.hypot(h[0]-end[0],h[1]-end[1])<=10 for h in hits[-2:])):
        end=None
    result["terminal_supported"]=bool(end)
    result["last30_same_location"]=int(sum(h is not None and np.hypot(h[0]-end[0],h[1]-end[1])<=8 for h in hits[-30:])) if end else 0
    segments=[]
    start=None
    for i,h in enumerate([*hits,None]):
        clear=h is not None and h[2]>=5 and i>=valid
        if start is not None:
            origin=hits[start]
            if not clear or np.hypot(h[0]-origin[0],h[1]-origin[1])>8:
                if i-start>=3:
                    portions=widths[valid:start+1]
                    # Match opening to the first clear view. Re-closing during withdrawal
                    # can otherwise hide the release in the median of a long stationary tail.
                    near=np.arange(max(valid,start-30),min(i,start+15))
                    closed_at=np.flatnonzero((np.maximum.accumulate(portions)-portions)>=5)
                    opening=0.
                    if len(closed_at):
                        first=valid+int(closed_at[0])
                        low=np.minimum.accumulate(widths[first:i])
                        near=near[near>=first]
                        if len(near):
                            opening=float(np.max(widths[near]-low[near-first]))
                    closure=float(np.max(np.maximum.accumulate(portions)-portions))
                    segments.append(dict(first=start,last=i-1,count=i-start,
                                         seconds=[round(float(times[k]),2) for k in (start,i-1)],
                                         open_mm=round(opening,1),close_mm=round(closure,1),
                                         ends_near_final=bool(end and np.hypot(origin[0]-end[0],origin[1]-end[1])<=12)))
                start=None
        if start is None and clear:
            start=i
    released=[s for s in segments if s["open_mm"]>=5 and s["close_mm"]>=5]
    matching=[s for s in released if s["ends_near_final"]]
    result["strongest_release"]=max(released,key=lambda s:s["count"],default=None)
    result["strongest_matching_release"]=max(matching,key=lambda s:s["count"],default=None)
    result["strongest_clear_at_end_position"]=max((s for s in segments if s["ends_near_final"]),key=lambda s:s["count"],default=None)
    result["last_detection_s"]=next((round(float(times[i]),2) for i in range(n-1,-1,-1) if hits[i] is not None),None)
    return result


def inspect(base,name):
    cv2.setNumThreads(1)
    ep=base/"recordings"/name
    rows=[json.loads(s) for s in (ep/"alignment.jsonl").read_text().splitlines()]
    paths=overhead(ep)
    if len(paths)!=len(rows) or len(rows)<30:
        raise ValueError("Insufficient aligned overhead images")
    ref=background(paths)
    rim=dish_outline(ref)
    own=rim is not None
    if rim is None:
        rim=dish_outline(background(overhead(base/"recordings/20261120_165700_4177037")))
    if rim is None:
        raise ValueError("Right dish boundary unavailable")
    mask=np.zeros(ref.shape[:2],"uint8")
    cv2.fillConvexPoly(mask,rim,1)
    mask=cv2.erode(mask,np.ones((5,5),"uint8"))
    x,y,w,h=cv2.boundingRect(rim)
    region=np.s_[y-22:y+h+22,x-22:x+w+22]
    local_mask=mask[region].astype(bool)
    ref_value=cv2.cvtColor(ref[region],cv2.COLOR_BGR2HSV)[:,:,2]
    hits=[block(cv2.imread(str(f))[region],ref_value,local_mask) for f in paths]
    widths=np.array([r["feedback"]["gripper_mm"] for r in rows])
    ns=np.array([r["sample_monotonic_ns"] for r in rows],dtype="i8")
    valid=next((i for i,r in enumerate(rows) if r["target"] is not None),len(rows))
    return dict(own_rim=own,rim_box=[x,y,w,h],**summarize(hits,widths,(ns-ns[0])/1e9,valid))


if __name__=="__main__":
    print(json.dumps(inspect(Path(sys.argv[1]),sys.argv[2])))
