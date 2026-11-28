"""Exercise current-edge and short-history preconditions with synthetic pixels only."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
from temporal import TemporalTarget

def check():
    shape=(480,640,3)
    plain=np.full(shape,[215,118,114],np.uint8)
    face=plain.copy()
    box=(276,312,150,144)
    x,y,w,h=box
    face[y:y+h,x:x+w]=(113,133,141)
    horizontal=plain.copy();horizontal[y:]=(113,133,141)
    color=lambda x:cv2.cvtColor(x,cv2.COLOR_LAB2BGR)
    def replay(current,history=3,gap=35_000_000):
        obs=TemporalTarget({})
        t=1_000_000_000
        for k in range(history):
            obs.observe(t+k*35_000_000,color(face),65.,box)
        stamp=t+max(0,history-1)*35_000_000+gap
        return obs.observe(stamp,color(current),65.,None)
    results=dict(visible_edges_with_history=replay(face),blank_with_history=replay(plain),
        single_edge_with_history=replay(horizontal),visible_edges_without_history=replay(face,0),
        expired_history=replay(face,gap=350_000_000))
    assert list(results.values())==[True,False,False,False,False],results
    return results
if __name__=='__main__':print(json.dumps(check()))
