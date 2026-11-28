"""Measure approach activation and projected reference disagreement in the older blind batch."""
import cv2
import numpy as np
from replay import models, read,here,work,load_record,endpoint,imaging,dump

def measure():
    cv2.setNumThreads(1)
    model=models()[1]
    prior=read(here/'archive.json')
    planned=read(here/'old_manifest.json')['unselected']
    gt=np.asarray(read(work/'offline_diagnostics/b1_vision_contrasts_20261124/final.json')['geometry']['table_map'])
    samples=[];tiles=[]
    for ep,item in sorted(prior['episodes'].items()):
        if not item['eligible']:continue
        rec=load_record(work/'recordings'/ep);rec['close']=item['close_frame']
        j=rec['close']-18
        end,_=endpoint(model,rec,rec)
        trace=model.trace['alignment'];uv=np.array(trace['block_uv'])
        rows=rec['rows']
        from replay import valid,at
        contact=next(i for i in range(max(0,rec['close']-6),len(rows)-2) if all(valid(v) and at(v,'feedback')[6]<=49 for v in rows[i:i+3]))
        dest=model.aligner.fk(at(rows[contact],'feedback'))
        samples.append(dict(episode=ep,novel=item['nearest_prior_mm']>=10,active=trace['applied'],estimated_gap_mm=trace['distance_mm'],
                            projected_to_expert_mm=float(np.linalg.norm(np.r_[uv,1.]@gt-dest))))
        if len(tiles)<3 and item['nearest_prior_mm']>=10:
            image=imaging(rec,j,'overhead')
            cv2.circle(image,tuple(uv.astype(int)),8,(240,190,0),2)
            tiles.append(cv2.resize(image,(320,240)))
    cv2.imwrite(str(here/'older_views.jpg'),np.hstack(tiles))
    dump('coverage.json',samples)
