"""Inspect archive frames used for overhead localization. No device access."""
from pathlib import Path
import sys,json
import numpy as np
import cv2
root=Path(__file__).resolve().parent
work,old,prev=map(Path,sys.argv[1:4])
sys.path.insert(0,str(prev))
import replay as rp
read=rp.read

def rows(group):
    manifest=read(prev/'old_manifest.json')
    eps=sorted(manifest['selected']) if group=='train' else sorted(read(prev/'archive.json')['episodes'])
    for ep in eps:
        r=rp.load_record(work/'recordings'/ep)
        seq=r['rows']
        c=next(i for i in range(1,len(seq)) if rp.valid(seq[i-1]) and rp.valid(seq[i]) and rp.at(seq[i-1],'target')[6]>55>=rp.at(seq[i],'target')[6])
        r['close']=c
        yield ep,r

def board(group):
    items=list(rows(group))
    meta=[]
    for offset in range(0,len(items),10):
        bands=[]
        for i,(ep,r) in enumerate(items[offset:offset+10],offset):
            c=r['close']; ticks=[0,max(0,(c-18)//2),max(0,c-18)]
            frames=[]
            for j in ticks:
                im=rp.imaging(r,j,'overhead')
                crop=im[214:450,96:440].copy()
                frames.append(cv2.resize(crop,(258,177),interpolation=cv2.INTER_AREA))
            band=np.hstack(frames)
            cv2.putText(band,f'{i} {ep[9:15]}',(4,14),0,.5,(255,255,255),1)
            bands.append(band)
            meta.append(dict(id=ep,idx=i,ticks=ticks))
        cv2.imwrite(str(root/f'{group}{offset}.jpg'),np.vstack(bands))
    (root/(group+'_index.json')).write_text(json.dumps(meta))

if __name__=='__main__':
    if len(sys.argv)>4:
        _,r=next(rows('train'))
        sel=read(work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json')['ranking'][0]
        new=rp.load_record(work/'batches/B1_20261123'/sel['source'])
        imgs=[rp.imaging(rec,j,'overhead') for rec,j in ((r,0),(r,r['close']-18),(new,0),(new,35))]
        cv2.imwrite(str(root/'color.jpg'),np.vstack([np.hstack(imgs[:2]),np.hstack(imgs[2:])]))
        for im in imgs:print('purple probe',np.median(cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[390:402,425:437],axis=(0,1)),flush=True)
        sys.exit(0)
    for group in ('test','train'):board(group)
    print('sheets complete')
