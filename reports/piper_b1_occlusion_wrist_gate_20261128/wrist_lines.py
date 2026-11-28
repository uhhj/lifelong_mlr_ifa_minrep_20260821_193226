"""Extract a wooden face from three intersecting visible lines in the wrist image."""
import cv2
import numpy as np
from collections import deque
from temporal import TemporalTarget
from b1_alignment_gate import wrist_centered

def segments(channel):
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(channel)[0]
    if raw is None:return [],[]
    sides=[];tops=[]
    lines=raw if raw.ndim==2 else raw[:,0,:]
    for x1,y1,x2,y2 in lines:
        dx=x2-x1;dy=y2-y1
        if abs(dy)>=16 and abs(dx)<=.48*abs(dy):
            k=dx/dy;sides.append((k,x1-k*y1,min(y1,y2),max(y1,y2)))
        elif abs(dx)>=23 and abs(dy)<=.45*abs(dx):
            k=dy/dx;tops.append((k,y1-k*x1,min(x1,x2),max(x1,x2)))
    return sides,tops

def face_candidates(frame):
    # Half-resolution working coordinates within the jaw workspace.
    small=cv2.resize(frame[190:480,138:562],(212,145),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(cv2.GaussianBlur(small,(3,3),0),cv2.COLOR_BGR2LAB)
    sides,tops=segments(lab[:,:,0])
    more,extra=segments(lab[:,:,1]);sides+=more;tops+=extra
    candidates=[]
    for slope,offset,start,end in tops:
        if end-start>115:continue
        left=[];right=[]
        for k,b,lo,hi in sides:
            y=(slope*b+offset)/(1-slope*k);x=k*y+b
            if not (8<y<122 and lo-11<=y<=lo+16 and hi-y>=15):continue
            if start-10<=x<=start+10:left.append((x,y,k,b,hi))
            if end-10<=x<=end+10:right.append((x,y,k,b,hi))
        for lx,ly,lk,lb,lh in left:
            for rx,ry,rk,rb,rh in right:
                w=rx-lx
                if not (39<=w<=104 and abs(lk-rk)<=.35):continue
                ymax=min(lh,rh,143)
                ytop=(ly+ry)/2
                if ymax-max(ly,ry)<min(26,.36*w):continue
                ymid=int((ymax+max(ly,ry))/2)
                xl=int(lk*ymid+lb+.22*w);xr=int(rk*ymid+rb-.22*w)
                if xl<2 or xr>=210 or xr-xl<12:continue
                patch=lab[ymid-4:ymid+5,xl:xr]
                L,A,B=np.median(patch,axis=(0,1))
                if not (32<=L<=193 and 115<=A<=149 and 128<=B<=157):continue
                left_x=lk*ymid+lb
                extent=ymax-ytop
                box=np.array([138+2*left_x,190+2*ytop,2*w,2*extent])
                score=(abs(lx-start)+abs(rx-end))*.15-min(ymax-ytop,w)*.5
                candidates.append((float(score),box))
    candidates.sort(key=lambda p:p[0])
    return candidates

def visible_face(frame):
    options=outline_candidates(frame)
    if not options:options=face_candidates(frame)
    if not options:return None
    box=options[0][1]
    if any(np.linalg.norm(b[:2]-box[:2])>30 for s,b in options[1:] if s<options[0][0]+10):return None
    return box

def outline_candidates(frame):
    # The visible front face may run off the bottom of the image. Use the top
    # and the two measured side boundaries, without requiring a closed contour.
    crop=cv2.resize(frame[190:480,138:562],(212,145),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(cv2.GaussianBlur(crop,(3,3),0),cv2.COLOR_BGR2LAB)
    colors=lab.astype(float);out=[]
    kernel=np.ones((3,3),np.uint8)
    for lo,hi in [((28,114,130),(163,147,153)),((24,117,128),(133,144,151)),((48,118,131),(190,149,159))]:
        mask=cv2.morphologyEx(cv2.inRange(lab,lo,hi),cv2.MORPH_CLOSE,kernel)
        n,ids,stats,_=cv2.connectedComponentsWithStats(mask)
        for k,(x,y,w,h,area) in enumerate(stats[1:],1):
            if not (36<=w<=118 and 24<=h<=118 and 5<y<124 and x>=3 and x+w<210 and area>=.50*w*h):continue
            comp=ids[y:y+h,x:x+w]==k
            xs=np.arange(int(w*.19),int(w*.81))
            if not comp[:,xs].any(0).all():continue
            tops=np.argmax(comp[:,xs],axis=0)
            slope,bias=np.polyfit(xs,tops,1)
            if abs(slope)>.5 or np.median(abs(tops-(slope*xs+bias)))>2.7:continue
            top=float(np.median(tops))
            first=max(2,int(max(tops)+3));last=min(h-2,int(top+.86*w))
            if last-first<16:continue
            ys=np.arange(first,last)
            band=comp[ys];usable=band.any(1)
            ys=ys[usable];band=band[usable]
            if len(ys)<16:continue
            left=np.argmax(band,axis=1);right=w-1-np.argmax(band[:,::-1],axis=1)
            fw=float(np.median(right-left+1))
            if not 38<=fw<=104 or np.percentile(abs(right-left+1-fw),80)>.15*fw:continue
            # Compare both sides of each observed boundary in the unthresholded image.
            def contrast(inner,outer):
                return float(np.mean(np.linalg.norm(inner-outer,axis=1)>=9))
            top_edge=contrast(colors[y+tops+3,x+xs],colors[y+tops-3,x+xs])
            left_edge=contrast(colors[y+ys,x+left+3],colors[y+ys,x+left-3])
            right_edge=contrast(colors[y+ys,x+right-3],colors[y+ys,x+right+3])
            if min(top_edge,left_edge,right_edge)<.52:continue
            mid=float(np.median((left+right)/2));height=h-top
            box=np.array([138+2*(x+mid-fw/2),190+2*(y+top),2*fw,2*height])
            score=-12*(top_edge+left_edge+right_edge)-min(height,fw)*.08
            out.append((score,box))
    return sorted(out,key=lambda r:r[0])

class WristEvidence(TemporalTarget):
    def __init__(self,calibration):
        super().__init__(calibration)
        self.corners=deque(maxlen=4);self.edge_stamp=0
        self.line_box=None
        self.wrist_goal=None
        self.uncertainty=0.
    def step(self,start,coarse,registered,gain,cap):
        if self.wrist_goal is None:return super().step(start,coarse,registered,gain,cap)
        displacement=self.wrist_goal-start
        distance=np.linalg.norm(displacement)
        length=min(cap,gain*max(0.,distance-self.uncertainty))
        return displacement*length/max(distance,1e-9)
    def observe(self,stamp,frame,opening,box):
        prior=super().observe(stamp,frame,opening,box)
        if not 0<stamp-self.edge_stamp<=70_000_000:self.corners.clear()
        self.edge_stamp=stamp
        face=visible_face(frame)
        self.line_box=face
        if face is None:
            self.corners.clear()
            return prior
        if self.corners and max(abs(face[:3]-self.corners[-1][:3]))>20:self.corners.clear()
        self.corners.append(face)
        if len(self.corners)<3 or not wrist_centered(face,opening):
            return prior
        self.visible_width=float(face[2])
        self.source='three_visible_sides'
        return True
