"""Motion labels use only expert feedback and a goal reference, in robot-base millimetres."""
import numpy as np
def classify(start,end,goal,opening,scale=1.):
    movement=end-start
    h=np.linalg.norm(movement[:2])
    d0=np.linalg.norm(start[:2]-goal);d1=np.linalg.norm(end[:2]-goal)
    along=float((goal-start[:2])@movement[:2])/max(h*d0,1e-9)
    clearance=min(12.,max(0.,(opening-35.)/2-2.))
    if d0<=60 and h>=3*scale and d0-d1>=2*scale and along>=.7:return 'correct'
    if max(d0,d1)<=clearance and h<=2*scale and -movement[2]>=max(3*scale,2*h):return 'descend'
    return 'other'
