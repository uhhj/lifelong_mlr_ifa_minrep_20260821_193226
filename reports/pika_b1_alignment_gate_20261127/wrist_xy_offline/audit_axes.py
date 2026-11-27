"""Check direction on independent matched starting views and expert contact locations."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from wrist_cue import table_direction
from wrist_target import top_face


def audit(tip, flange):
    root=Path(__file__).resolve().parent
    camera=json.loads((root/'wrist_camera.json').read_bytes())
    labels=json.loads((root/'refs.json').read_bytes())
    with np.load(root/'private_refs.npz') as reference:
        q=reference['q']; images=reference['im'][:,0,1]; xyz=reference['xy']
    usable=[]
    for i,e in enumerate(labels):
        if e['train']:continue
        face=top_face(images[i])
        if face is None:continue
        rot=flange(np.rad2deg(q[i,0]))[:3,:3]
        usable.append((i,face,rot))
    candidates=[]
    for (i,fi,ri),(j,fj,rj) in combinations(usable,2):
        angle=np.rad2deg(np.arccos(np.clip((np.trace(ri.T@rj)-1)/2,-1,1)))
        distance=np.linalg.norm(xyz[i,0]-xyz[j,0])
        if angle>3 or distance>6 or abs(np.log(fi[2]/fj[2]))>.12:continue
        # Pair episodes by initial tool pose only, before consulting expert endpoints.
        candidates.append((angle+distance/2,i,j,fi,fj,ri))
    out={}
    for axis,side in enumerate(('right','bottom')):
        used=set(); comparisons=[]
        for _,i,j,fi,fj,rot in sorted(candidates,key=lambda r:r[0]):
            delta_px=float(fj[axis]-fi[axis])
            if abs(delta_px)<10 or i in used or j in used:continue
            dx=table_direction(rot,camera,'right');dy=table_direction(rot,camera,'bottom')
            if dx is None or dy is None:continue
            observed=(xyz[j,-1,:2]-xyz[j,0,:2])-(xyz[i,-1,:2]-xyz[i,0,:2])
            axes=np.column_stack((dx,dy))
            delta_mm=float(np.linalg.solve(axes,observed)[axis])
            used.update((i,j))
            comparisons.append(dict(episode_a=labels[i]['episode'],episode_b=labels[j]['episode'],
                                    delta_px=round(delta_px,3),expert_mm=round(delta_mm,3),
                                    informative=abs(delta_mm)>=3,agrees=delta_px*delta_mm>0))
        useful=[x for x in comparisons if x['informative']]
        out[side]=dict(matched=len(comparisons),informative=len(useful),agree=sum(x['agrees'] for x in useful),pairs=comparisons)
    (root/'axis_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print('independent direction checks',[(k,v['informative'],v['agree']) for k,v in out.items()],flush=True)
