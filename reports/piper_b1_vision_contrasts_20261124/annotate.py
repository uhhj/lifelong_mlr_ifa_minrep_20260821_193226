"""Extract stationary block observations for review before inspecting any predictions."""
from decimal import Decimal
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def images(folder,role):
    return sorted((folder/"episode0/camera/color"/role).glob("*.jpg"),key=lambda f:Decimal(f.stem))


def vectors(rows,key):
    return np.asarray([r[key]["joints_rad"]+[r[key]["gripper_mm"]] for r in rows],dtype=np.float32)


def locate(work,out):
    spatial=work/"offline_diagnostics/b1_spatial_ablation_20261121_2034"
    sys.path.insert(0,str(spatial))
    from block_track import detect
    selection=json.loads((work/"offline_diagnostics/b1_ode_lerobot_20261123/selection.json").read_bytes())
    training=set(selection["selected_ids"])
    candidates=sorted(selection["ranking"],key=lambda r:r["id"])
    records=[]
    groups=[[] for _ in range(4)]
    for i,r in enumerate(candidates):
        folder=work/"batches/B1_20261123"/r["source"]
        files=images(folder,"overhead")[r["first"]:]
        rows=[json.loads(line) for line in (folder/"alignment.jsonl").read_bytes().splitlines()][r["first"]:]
        cmd=vectors(rows,"target")
        state=vectors(rows,"feedback")
        hits=np.flatnonzero((cmd[:-1,6]>55)&(cmd[1:,6]<=55))+1
        cross=int(hits[0]) if len(hits) and cmd[0,6]>55 else None
        grip=np.flatnonzero((state[:-2,6]<=49)&(state[1:-1,6]<=49)&(state[2:,6]<=49))
        grip=grip[grip>=max(3,(cross or 0)-6)]
        contact=int(grip[0]) if len(grip) else None
        options=[]
        stop=max(1,min((cross if cross is not None else 32)-5,52))
        ticks=sorted(set([0,*range(2,stop,3)]))
        for j in ticks:
            im=cv2.imread(str(files[j]))
            pos=detect(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            if pos is not None:options.append((j,np.array(pos)))
        center=None
        if options:
            xy=np.stack([pos for j,pos in options])
            dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
            idx=int(np.argmax((dist<8).sum(1)))
            keep=dist[idx]<8
            center=np.median(xy[keep],axis=0)
            j=options[np.argmin(np.linalg.norm(xy-center,axis=1))][0]
        else:j=ticks[len(ticks)//2]
        record=dict(index=i,episode=r["id"],train=r["id"] in training,frame=j,
                    uv=None if center is None else center.tolist(),close=cross,contact=contact,
                    candidates=len(options),support=int(keep.sum()) if options else 0)
        records.append(record)
        image=cv2.imread(str(files[j]))
        if center is not None:
            cv2.circle(image,tuple(np.rint(center).astype(int)),14,(245,255,10),1)
        tile=image[245:455,105:430]
        tile=cv2.resize(tile,(244,158),interpolation=cv2.INTER_AREA)
        label=np.full((21,244,3),255,np.uint8)
        cv2.putText(label,f"{i:02d}  {('T' if record['train'] else 'E')}  {j}  n={record['support']}",(3,15),0,.45,(10,10,10),1)
        groups[i//21].append(np.vstack([label,tile]))
    for i,tiles in enumerate(groups):
        while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out/f"view{i}.jpg"),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,len(tiles),3)]))
    (out/"points.json").write_text(json.dumps(records))
    print(json.dumps({"episodes":len(records),"located":sum(r["uv"] is not None for r in records)}))


if __name__=="__main__":
    locate(*map(Path,sys.argv[1:3]))
