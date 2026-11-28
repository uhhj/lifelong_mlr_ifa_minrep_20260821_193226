"""Estimate the stationary cube in table coordinates and request a bounded eight-step XY move."""
from collections import deque
import cv2
import numpy as np


def wrist_measurement(frame, tip, rotation, coarse, cal):
    axes = rotation @ np.asarray(cal['R'])
    origin = tip + rotation @ np.asarray(cal['offset'])
    center = (np.r_[coarse, cal['z']] - origin) @ axes
    if center[2] <= 30:
        return None
    focal = cal['f'] / 2
    projected = np.array([160., 120.]) + focal * center[:2] / center[2]
    side = 35 * focal / center[2]
    if not 12 <= side <= 142:
        return None
    small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small,(3,3),0),cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    edges = cv2.Canny(light, 15, 40) | cv2.Canny(a, 10, 25) | cv2.Canny(b, 10, 25)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    wood = cv2.inRange(lab,(38,117,128),(220,151,165))
    contours = list(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.extend(cv2.findContours(cv2.morphologyEx(wood,cv2.MORPH_OPEN,kernel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
    candidates = []
    for c in contours:
        (u,v),(w,h),angle = cv2.minAreaRect(c)
        if min(w,h) < .56*side or max(w,h)>1.6*side or min(w,h)<.65*max(w,h):
            continue
        corners=cv2.boxPoints(((u,v),(w,h),angle))
        if np.any(corners.min(0)<[2,40]) or np.any(corners.max(0)>[317,237]):
            continue
        fill=cv2.contourArea(c)/(w*h)
        if fill < .77 or np.linalg.norm(projected-[u,v])>1.15*side:
            continue
        x,y,cw,ch=cv2.boundingRect(c)
        yy,xx=int(round(v)),int(round(u))
        rr=max(3,int(.21*min(w,h)))
        color=np.median(lab[yy-rr:yy+rr+1,xx-rr:xx+rr+1],axis=(0,1))
        if not (42<=color[0]<=210 and 120<=color[1]<=146 and 129<=color[2]<=160):
            continue
        ray=axes@np.array([(u-160)/focal,(v-120)/focal,1.])
        if ray[2]>=-.35:
            continue
        xy=origin[:2]+ray[:2]*((cal['z']-origin[2])/ray[2])
        separation=np.linalg.norm(xy-coarse)
        if separation<=25:
            candidates.append((separation,xy))
    if not candidates:
        return None
    candidates.sort(key=lambda c:c[0])
    if any(np.linalg.norm(xy-candidates[0][1])>10 for _,xy in candidates[1:]):
        return None
    return candidates[0][1]


class RelativeTarget:
    def __init__(self, calibration):
        self.calibration=calibration
        self.observations=deque(maxlen=8)
        self.last_stamp=0
        self.target=None
        self.centered=False

    def update(self,stamp,wrist,tip,rotation,coarse,opening):
        if not 0<stamp-self.last_stamp<=70_000_000:
            self.observations.clear()
        self.last_stamp=stamp
        xy=wrist_measurement(wrist,tip,rotation,coarse,self.calibration)
        if xy is not None:
            self.observations.append((stamp,xy))
        recent=[p for t,p in self.observations if 0<=stamp-t<=300_000_000 and np.linalg.norm(coarse-p)<=25]
        self.target=None
        self.centered=False
        if len(recent)>=3:
            median=np.median(recent,axis=0)
            if np.max(np.linalg.norm(np.asarray(recent)-median,axis=1))<=10:
                self.target=median
                clearance=min(12.,max(0.,(opening-35.)/2-2.))
                self.centered=xy is not None and np.linalg.norm(tip[:2]-median)<=clearance

    def step(self,start,coarse,registered,gain,cap):
        # Coarse table calibration and pose-corrected estimate respectively provide
        # displacement sensitivity and compensation for systematic registration error.
        goal=.55*coarse+.45*registered
        if self.target is not None:
            goal=.4*goal+.6*self.target
        delta=(goal-start)*gain
        return delta*min(1.,cap/max(np.linalg.norm(delta),1e-9))
