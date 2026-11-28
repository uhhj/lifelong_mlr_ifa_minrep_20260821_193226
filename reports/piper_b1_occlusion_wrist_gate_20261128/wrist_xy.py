"""Tool-mounted camera position estimate, usable for approach guidance only."""
from collections import deque
import cv2
import numpy as np
from b1_alignment_gate import wrist_block

def image_box(frame):
    box=wrist_block(frame)
    if box is not None:return np.array(box,dtype=float)
    small=cv2.resize(frame[174:480,120:580],(230,153),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(cv2.GaussianBlur(small,(3,3),0),cv2.COLOR_BGR2LAB)
    choices=[];kernel=np.ones((3,3),np.uint8)
    for lo,hi in [((32,116,131),(167,148,154)),((22,117,130),(126,144,151)),((46,115,129),(190,145,154))]:
        mask=cv2.inRange(lab,lo,hi)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        components=cv2.connectedComponentsWithStats(mask)[2][1:]
        for x,y,w,h,area in components:
            if not (4<x and x+w<226 and y>2 and 31<=w<=105 and h>=min(30,.55*w) and h<=1.5*w and area>=.64*w*h):continue
            if h<.59*w and y+h<151:continue
            choices.append(np.array([120+2*x,174+2*y,2*w,2*h],float))
    if not choices:return None
    centers=np.array([b[:2]+b[2:]/2 for b in choices])
    if np.max(np.linalg.norm(centers-centers[0],axis=1))>20:return None
    return np.median(choices,axis=0)

def features(frame,tip,rotation,cal):
    box=image_box(frame)
    axes=rotation@np.array(cal['R'])
    flat=axes[:2,:2]
    if box is None or abs(np.linalg.det(flat))<.45:return None
    x,y,w,h=box
    u=x+w/2;v=y+.5*min(1.1*w,max(h,.82*w))
    feat=np.array([1.,(u-355)*35/w,(v-405)*35/w,35*(145/w-1),tip[2]/20,15*axes[2,0],15*axes[2,1]])
    return feat,flat

class WristPosition:
    def __init__(self,fits):
        self.fits=fits;self.view=None;self.last=0;self.points=deque(maxlen=5)
    def update(self,stamp,view,frame,tip,rotation,cal):
        if self.view!=view or not 0<stamp-self.last<=70_000_000:self.points.clear()
        self.last=stamp;self.view=view
        if view is None:return None
        fit=self.fits[view]
        info=features(frame,tip,rotation,cal)
        if info is None:return None
        f,b=info
        if np.any(f<fit['lo']) or np.any(f>fit['hi']):return None
        pred=tip[:2]+b@(f@np.array(fit['coef']))
        if np.linalg.norm(pred-tip[:2])>60:return None
        self.points.append((stamp,pred))
        recent=[p for t,p in self.points if 0<=stamp-t<=170_000_000]
        if len(recent)<3:return None
        xy=np.median(recent,axis=0)
        if max(np.linalg.norm(p-xy) for p in recent)>9:return None
        return xy
