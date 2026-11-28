"""Offline B1 registration and causal partial-face confirmation in the raw wrist view."""
from collections import deque
import cv2
import numpy as np
from b1_alignment_gate import wrist_centered
from relative_xy import RelativeTarget

def signature(frame):
    # The upper background is outside the tabletop block workspace.
    crop = cv2.resize(frame[8:150,32:608], (24,8), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(float)
    return (lab - np.median(lab, axis=(0,1))).ravel()

class Registration:
    def __init__(self, data):
        self.reference = np.array(data['signatures'])
        self.old_matrix = np.array(data['archive_matrix'])
        self.limit = data['distance_limits']
        self.close_envelopes = data['close_envelopes']

    def locate(self, frame, uv, current):
        delta = self.reference - signature(frame)
        dist = np.sqrt(np.mean(delta**2, axis=1))
        choice = int(np.argmin(dist))
        known = dist[choice] <= self.limit[choice] and dist[choice] + 2 < dist[1-choice]
        matrix = (self.old_matrix, current)[choice]
        return np.r_[uv,1.] @ matrix, choice if known else None

def edge_data(frame):
    small = cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small,(3,3),0),cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = np.max(abs(cv2.Sobel(lab,cv2.CV_32F,1,0)),axis=2)
    gy = np.max(abs(cv2.Sobel(lab,cv2.CV_32F,0,1)),axis=2)
    return lab,gx,gy

def supported_box(data, expected):
    """Follow the top and at least one side. Only measure edges visible in this frame."""
    lab,gx,gy=data
    x,y,w,h=np.asarray(expected,dtype=float)/2
    x,y,w,h=map(round,(x,y,w,h))
    if min(w,h)<10 or x<14 or y<14 or x+w>=306 or y>224:
        return None
    a,b=int(x+.2*w),int(x+.8*w)
    top=gy[y-9:y+10,a:b]
    scores=np.mean(top>=32,axis=1)
    iy=int(np.argmax(scores))
    if scores[iy]<.54:return None
    yt=y-9+iy
    t=yt+int(.16*h);bt=min(236,yt+int(.8*h))
    if bt-t<12:return None
    sides=[]
    for u in (x,x+w):
        strip=gx[t:bt,u-8:u+9]
        vals=np.mean(strip>=28,axis=0)
        j=int(np.argmax(vals))
        sides.append(u-8+j if vals[j]>=.49 else None)
    left,right=sides
    if left is None and right is None:return None
    if left is None:left=right-w
    if right is None:right=left+w
    if not .8*w <= right-left <= 1.2*w:return None
    patch=lab[t:bt,left+int(.25*w):right-int(.25*w)]
    if not patch.size:return None
    L,A,B=np.median(patch,axis=(0,1))
    if not (32<=L<=180 and 117<=A<=149 and 130<=B<=155):return None
    return np.array([2*left,2*yt,2*(right-left),2*h],float)

class TemporalTarget(RelativeTarget):
    def __init__(self,calibration):
        super().__init__(calibration)
        self.faces=deque(maxlen=5)
        self.source=None
        self.archive_view=False
        self.visible_width=None

    def observe(self,stamp,frame,opening,box):
        if self.faces and not 0 < stamp-self.faces[-1][0] <= 100_000_000:
            self.faces.clear()
        while self.faces and stamp-self.faces[0][0]>250_000_000:self.faces.popleft()
        previous=[v for _,v in self.faces]
        data=edge_data(frame)
        direct=supported_box(data,box) if box is not None else None
        supported=None
        if len(previous)>=2 and np.max(abs(previous[-1][:2]-previous[-2][:2]))<=26:
            expected=previous[-1].copy()
            expected[:2]+=np.clip(previous[-1][:2]-previous[-2][:2],-12,12)
            supported=supported_box(data,expected)
        if direct is not None:
            if previous and np.max(abs(direct[:2]-previous[-1][:2]))>32:
                self.faces.clear()
                supported=None
            self.faces.append((stamp,direct))
        self.source=None
        self.visible_width=None
        if supported is None:return False
        self.visible_width=float(supported[2])
        self.source='recent_face_current_edges'
        return bool(wrist_centered(supported,opening))

    def step(self,start,coarse,registered,gain,cap):
        if not self.archive_view:return super().step(start,coarse,registered,gain,cap)
        # The residual learned in the newer view is defined in that registration only.
        goal=coarse if self.target is None else .4*coarse+.6*self.target
        delta=(goal-start)*gain
        return delta*min(1.,cap/max(np.linalg.norm(delta),1e-9))

