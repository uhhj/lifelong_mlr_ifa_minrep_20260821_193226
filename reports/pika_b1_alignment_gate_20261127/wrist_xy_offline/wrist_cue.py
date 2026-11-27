"""Estimate signed, one-sided image offsets from a visible wood/paper boundary."""
import cv2
import numpy as np


def visible_edges(lab):
    red=(lab[:,:,1].astype(float)>142)&(lab[:,:,1].astype(float)-lab[:,:,2]>4)
    red[:70]=False
    b=lab[:,:,2]
    kernel=np.ones((3,3),np.uint8)
    records=[]
    for low,high in [((20,121,131),(200,143,158)),((25,117,133),(195,146,158))]:
        wood=cv2.inRange(lab,low,high)
        wood[:70]=0
        wood=cv2.morphologyEx(wood,cv2.MORPH_OPEN,kernel)
        _,labels,stats,_=cv2.connectedComponentsWithStats(wood,8)
        for label,(x,y,w,h,area) in enumerate(stats[1:],1):
            if not (18<=w<=120 and 12<=h<=110 and area>=300 and area>=w*h*.35):continue
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==label)
            cols=np.unique(xx)
            if len(cols)<18:continue
            top=np.full(w,h);np.minimum.at(top,xx,yy)
            # Accept the upper edge only where a red strip lies immediately
            # above visible wood; wood, dark fingers and exposed gray desk differ.
            pix_y=top[cols]+y
            px=cols+x
            above=np.stack([red[pix_y-gap,px] for gap in (2,3,4,5,6)])
            supported=above.mean(axis=0)>=.6
            if supported.sum()<14:continue
            su,sv=px[supported],pix_y[supported]
            if np.ptp(su)<15:continue
            slope,intercept=np.polyfit(su,sv,1)
            if abs(slope)>.5 or np.median(abs(sv-(slope*su+intercept)))>2.5:continue
            # Reject floating chroma specks and a thin border at the edge of paper.
            yprobe=np.minimum(sv+7,239)
            inface=labels[yprobe,su]==label
            if inface.mean()<.65 or np.median(b[yprobe,su])<132:continue
            records.append((int(2*su.min()),int(2*su.max()),float(2*np.median(sv)),int(2*x),int(2*(x+w)),int(2*(y+h)),int(supported.sum())))
    return records


def side_face(lab):
    """Check both ends of vertical wood chords on red paper at a side border."""
    light, a, b = cv2.split(lab.astype(np.int16))
    red = (a > 143) & (a - b > 4)
    wood = (light > 46) & (light < 225) & (a >= 118) & (a <= 152) & (b >= 127) & (b <= 167) & ~red
    red_sum = np.pad(np.cumsum(red,axis=0,dtype=np.int32),((1,0),(0,0)))
    wood_sum = np.pad(np.cumsum(wood,axis=0,dtype=np.int32),((1,0),(0,0)))
    directions = set()
    for side, col in [('right', np.arange(232,320,3)), ('left', np.arange(0,129,3))]:
        # Pair start/end transitions column by column using red sentinels.
        scan = np.pad(red[70:218,col].T.astype(np.int8),((0,0),(1,1)),constant_values=1)
        changes = np.diff(scan,axis=1)
        columns, starts = np.where(changes==-1)
        _, ends = np.where(changes==1)
        x, top, bottom = col[columns], starts+70, ends+70
        span = bottom-top
        keep = (top>=80)&(top<=155)&(span>=35)&(span<=107)&(bottom<216)
        x,top,bottom = x[keep],top[keep],bottom[keep]
        keep = ((red_sum[top,x]-red_sum[top-5,x]>=4)
                &(red_sum[bottom+5,x]-red_sum[bottom,x]>=4)
                &(wood_sum[bottom-3,x]-wood_sum[top+3,x]>=.88*(bottom-top-6)))
        group=np.column_stack((x[keep],top[keep],bottom[keep]))
        if len(group)<9:
            continue
        top,end=np.median(group[:,1:],axis=0)
        group=group[(abs(group[:,1]-top)<=8)&(abs(group[:,2]-end)<=10)]
        if len(group)<9 or np.ptp(group[:,0])<24:
            continue
        # Verify that this same face does not extend back into the jaw opening.
        inside=slice(199,219) if side=='right' else slice(138,158)
        near=wood[int(top+8):int(end-8),inside]
        depth=slice(int(top+8),int(end-8))
        if near.size and near.mean()<.5 and (np.median(b[depth,inside])<127 or red[depth,inside].mean()>.5):
            directions.add(side)
    return directions


def signed_cue(wrist):
    lab=cv2.cvtColor(cv2.resize(wrist,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2LAB)
    votes=set()
    for u0,u1,v,left,right,bottom,support in visible_edges(lab):
        mid=(u0+u1)/2
        if bottom-v>=65 and 155<=v<=338:
            if left>=443 and u0>=450: votes.add('right')
            if right<=265 and u1<=260: votes.add('left')
        # For a face cut by the bottom of the view, its visible upper edge
        # bounds its depth even though its center cannot be measured.
        if 392<=v<=454 and bottom>=476 and 285<=mid<=444 and u1-u0>=38 and support>=18:
            votes.add('bottom')
    votes.update(side_face(lab))
    return next(iter(votes)) if len(votes)==1 else None


class ConfirmedCue:
    def __init__(self):
        self.last_stamp = 0
        self.side = None
        self.count = 0

    def observe(self, stamp, side):
        consecutive = 0 < stamp - self.last_stamp <= 50_000_000
        self.count = self.count + 1 if consecutive and side and side == self.side else int(bool(side))
        self.last_stamp, self.side = stamp, side
        return side if self.count >= 3 else None


def table_direction(rotation, camera, side):
    """Horizontal displacement toward increasing wrist column/row at the jaw."""
    axes = rotation @ np.asarray(camera['R'])
    ray = axes @ np.array([35 / camera['f'], 135 / camera['f'], 1.])
    if ray[2] > -.4:
        return None
    mapping = axes[:2, :2] - np.outer(ray[:2] / ray[2], axes[2, :2])
    if np.linalg.cond(mapping) > 3:
        return None
    unit = mapping[:, int(side == 'bottom')]
    return (-1 if side == 'left' else 1) * unit / np.linalg.norm(unit)


def modified_step(step, mode, rotation, camera, cap):
    """Change the eight-step endpoint only on sustained wrist evidence."""
    norm = np.linalg.norm(step)
    if mode == 'centered':
        return step * min(1., .5 / max(norm, 1e-12))
    if mode not in ('left', 'right', 'bottom'):
        return step
    direction = table_direction(rotation, camera, mode)
    if direction is None:
        return step
    # On strong disagreement, remove only the contrary component. This avoids
    # driving toward a possibly touching finger on a partly visible block.
    along = float(step @ direction)
    if along < -2.:
        return step - along * direction
    addition = np.clip(2. - along, 0., 2.)
    revised = step + addition * direction
    return revised * min(1., cap / max(np.linalg.norm(revised), 1e-12))
