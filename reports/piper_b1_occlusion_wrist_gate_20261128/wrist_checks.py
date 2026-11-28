"""Exercise visible edges and independent guidance with synthetic frames only."""
import cv2
import numpy as np
from wrist_lines import WristEvidence,visible_face
def run():
    def frame(kind):
        im=np.full((480,640,3),[188,124,116],np.uint8)
        if kind=='wood':im[315:468,280:432]=(101,130,140)
        elif kind=='yellow':im[315:468,280:432]=(182,124,170)
        elif kind=='boundary':im[315:]=(101,130,140)
        return cv2.cvtColor(im,cv2.COLOR_LAB2BGR)
    w=WristEvidence({});dt=33_000_000
    states=[w.observe((i+1)*dt,frame('wood'),65,None) for i in range(3)]
    assert states==[False,False,True] and w.source=='three_visible_sides',states
    for i,k in enumerate(('blank','boundary','yellow'),4):
        assert visible_face(frame(k)) is None
        assert not w.observe(i*dt,frame(k),65,None)
    assert not w.observe(25*dt,frame('wood'),65,None)
    return ['independent current edges','three stable observations','blank image','single edge','yellow paper','history gap']
if __name__=='__main__':print(run())
