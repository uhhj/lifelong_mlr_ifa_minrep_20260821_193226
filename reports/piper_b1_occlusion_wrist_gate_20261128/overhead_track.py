"""Confirm a stationary tabletop target before accepting a heatmap relocation."""
from collections import deque
import cv2
import numpy as np

def visible_square(frame, proposal):
    """Require an actual square boundary and wooden interior near the heatmap peak."""
    u,v=np.rint(proposal).astype(int)
    x=max(0,u-46);y=max(0,v-46)
    crop=frame[y:min(frame.shape[0],v+47),x:min(frame.shape[1],u+47)]
    if min(crop.shape[:2])<24:return None
    lab=cv2.cvtColor(cv2.GaussianBlur(crop,(3,3),0),cv2.COLOR_BGR2LAB)
    light,a,b=cv2.split(lab)
    edges=cv2.Canny(light,20,55)|cv2.Canny(a,12,36)|cv2.Canny(b,12,36)
    kernel=np.ones((3,3),np.uint8)
    edge=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)
    wood=cv2.inRange(lab,(42,111,126),(216,153,166))
    contours=list(cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
    contours+=list(cv2.findContours(cv2.morphologyEx(wood,cv2.MORPH_OPEN,kernel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
    candidates=[]
    for c in contours:
        (cx,cy),(w,h),angle=cv2.minAreaRect(c)
        if min(w,h)<19 or max(w,h)>53 or min(w,h)<.62*max(w,h) or cv2.contourArea(c)<.69*w*h:continue
        corners=cv2.boxPoints(((cx,cy),(w,h),angle))
        if (corners.min(0)<2).any() or (corners.max(0)>[crop.shape[1]-3,crop.shape[0]-3]).any():continue
        xx,yy=round(cx),round(cy)
        center=np.array([cx+x,cy+y])
        gap=np.linalg.norm(center-proposal)
        if gap>21:continue
        interior=wood[yy-5:yy+6,xx-5:xx+6]
        if np.mean(interior>0)<.88:continue
        candidates.append((gap,center))
    if not candidates:return None
    candidates.sort(key=lambda r:r[0])
    if any(np.linalg.norm(c-candidates[0][1])>12 for _,c in candidates[1:]):return None
    return candidates[0][1]

class StationaryCube:
    def __init__(self):
        self.candidates=deque(maxlen=3)
        self.anchor=None
        self.view=None
        self.last_stamp=0
        self.confirmed_at=0

    def update(self,stamp,frame,proposal,view):
        if self.view!=view or not 0<stamp-self.last_stamp<=70_000_000:
            self.candidates.clear();self.anchor=None
        self.view=view;self.last_stamp=stamp
        if view is None:return None,'unknown_view'
        square=visible_square(frame,proposal)
        if square is not None:
            if self.candidates and np.linalg.norm(square-self.candidates[-1])>8:self.candidates.clear()
            self.candidates.append(square)
            if len(self.candidates)==3:
                self.anchor=np.median(self.candidates,axis=0);self.confirmed_at=stamp
                return self.anchor.copy(),'visible'
        else:self.candidates.clear()
        if self.anchor is not None and stamp-self.confirmed_at<=2_000_000_000:
            return self.anchor.copy(),'held'
        self.anchor=None
        return None,'unobserved'
