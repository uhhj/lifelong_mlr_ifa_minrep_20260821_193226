"""Rebuild the private frame selection from the previously verified B1 manifest."""
import json
from pathlib import Path
import sys
import zipfile


def sample(work,out):
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    sys.path.insert(0,str(prior))
    from annotate import images
    from assess import reviewed
    records=[]
    with zipfile.ZipFile(out/'frames.zip','w') as archive:
        for r in reviewed(work,prior):
            frames=images(r['folder'],'overhead')[r['first']:]
            near=r['close'] if r['close'] is not None else r['contact']
            phases=dict(review=r['frame'],approach=max(0,near-18),lift=min(near+35,len(frames)-1),
                        pickup=r['contact'],release=len(frames)-1)
            for phase,j in phases.items():
                key=f"{r['index']:02d}_{phase}"
                archive.write(frames[j],key+'.jpg')
                records.append(dict(key=key,index=r['index'],episode=r['episode'],
                                    train=r['train'],j=j,phase=phase,uv=r['uv']))
    (out/'frames.json').write_text(json.dumps(records))


if __name__=='__main__':
    sample(*map(Path,sys.argv[1:3]))
