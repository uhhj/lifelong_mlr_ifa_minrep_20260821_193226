"""Check acquisition, rejection, expiry and reacquisition with synthetic frames."""
import cv2
import numpy as np
from overhead_track import StationaryCube

def check():
    def frame(center):
        lab=np.full((480,640,3),[190,124,120],np.uint8)
        if center is not None:
            x,y=center;lab[y-15:y+16,x-15:x+16]=[108,128,141]
        return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    tracker=StationaryCube();stamp=0
    def tick(im,peak,delta=33_000_000,view=0):
        nonlocal stamp
        stamp+=delta
        return tracker.update(stamp,im,np.asarray(peak),view)
    p=(300,320);q=(180,280)
    for _ in range(3):anchor,state=tick(frame(p),p)
    assert state=='visible' and np.linalg.norm(anchor-p)<2
    blank=frame(None)
    for _ in range(32):
        anchor,state=tick(blank,q);assert state=='held' and np.linalg.norm(anchor-p)<2
    for _ in range(32):anchor,state=tick(blank,q)
    assert anchor is None and state=='unobserved'
    for _ in range(3):anchor,state=tick(frame(q),q)
    assert state=='visible' and np.linalg.norm(anchor-q)<2
    assert tick(blank,p,delta=120_000_000)[0] is None
    for _ in range(3):anchor,state=tick(frame(p),p)
    assert tick(blank,q,view=None)[0] is None
    return ['visible acquisition','occlusion rejects distant heatmap peak','expiry','new visible placement','sample gap','unknown scene']
if __name__=='__main__':print(check())
