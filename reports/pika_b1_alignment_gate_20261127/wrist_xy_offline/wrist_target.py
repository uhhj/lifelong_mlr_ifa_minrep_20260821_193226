"""Locate a complete wooden top face in the B1 wrist view."""
import cv2
import numpy as np


def top_face(image):
    im=cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA) if image.shape[:2]!=(240,320) else image
    lab=cv2.cvtColor(cv2.GaussianBlur(im,(3,3),0),cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    edges=(cv2.Canny(l,12,35)|cv2.Canny(a,9,22)|cv2.Canny(b,8,20))
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    proposals=[]
    wood=cv2.inRange(lab,(40,120,134),(210,146,164))
    wood=cv2.morphologyEx(wood,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours=list(cv2.findContours(wood,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.extend(cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
    for contour in contours:
        rect=cv2.minAreaRect(contour)
        (cx,cy),(w,h),_=rect
        if not (20<cx<305 and 55<cy<222 and 24<=min(w,h)<=95 and max(w,h)<=103 and min(w,h)/max(w,h)>.70):continue
        fill=cv2.contourArea(contour)/(w*h)
        if fill<.80:continue
        corners=cv2.boxPoints(rect)
        if np.any(corners.min(axis=0)<2) or np.any(corners.max(axis=0)>[317,237]):continue
        inner=np.zeros(im.shape[:2],np.uint8)
        cv2.fillConvexPoly(inner,np.rint([cx,cy]+.78*(corners-[cx,cy])).astype('int32'),1)
        inside=lab[inner!=0]
        median=np.median(inside,axis=0)
        if not(38<=median[0]<=193 and 121<=median[1]<=144 and 133<=median[2]<=160):continue
        ring=np.zeros_like(inner)
        cv2.fillConvexPoly(ring,np.rint([cx,cy]+1.31*(corners-[cx,cy])).astype('int32'),1)
        cv2.fillConvexPoly(ring,np.rint([cx,cy]+1.03*(corners-[cx,cy])).astype('int32'),0)
        if np.mean(a[ring!=0]>145)<.10:continue
        proposals.append((fill,cx,cy,w,h))
    if not proposals:return None
    proposals.sort(reverse=True)
    _,cx,cy,w,h=proposals[0]
    if any(np.hypot(x-cx,y-cy)>min(w,h)*.4 for _,x,y,_,_ in proposals[1:]):return None
    return np.array([2*cx,2*cy,2*np.sqrt(w*h)])
