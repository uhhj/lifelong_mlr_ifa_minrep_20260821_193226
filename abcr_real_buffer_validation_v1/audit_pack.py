"""Blinded AI semantic audit images. No action label, matching score or gate shown."""
import argparse
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw
from .common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    pairs=read(a.out/'AUDIT_SELECTION_PRIVATE.json')['pairs'];dest=a.out/'semantic_audit';dest.mkdir(exist_ok=True);blinded=[];canvases=[]
    for r in pairs:
        panel=Image.new('RGB',(1024,310),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),r['pair_id'],fill='black')
        for side,x in [('new',0),('old',512)]:
            lang=r[side+'_language'];target=lang.replace('pick up the ','').replace(' and place it in the basket','')
            draw.text((x+4,18),side.upper()+': '+target+' | demo%d frame%d'%(r[side+'_demo'],r[side+'_t']),fill='black')
            with h5py.File(r[side+'_hdf5'],'r') as h:
                for k,view in enumerate(['agentview_rgb','eye_in_hand_rgb']):
                    pixels=h['data/demo_%d/obs/%s'%(r[side+'_demo'],view)][r[side+'_t']]
                    im=Image.fromarray(np.flipud(pixels).copy()).resize((256,256))
                    panel.paste(im,(x+k*256,46))
        panel.save(dest/(r['pair_id']+'.png'));canvases.append(panel)
        blinded.append({k:r[k] for k in ['pair_id','new_language','old_language','new_demo','old_demo','new_t','old_t']})
    for i in range(0,len(canvases),4):
        sheet=Image.new('RGB',(1024,310*len(canvases[i:i+4])),'white')
        for j,p in enumerate(canvases[i:i+4]):sheet.paste(p,(0,j*310))
        sheet.save(dest/('board_%02d.png'%(i//4+1)))
    write(dest/'BLINDED_MANIFEST.json',dict(records=blinded,reviewer='AI, not human',criterion=PROTOCOL['semantic'],images='raw two views, display-only vertical flip for OpenGL, no feature scores/actions/gate outcomes'))
    print('AUDIT_PACK',len(pairs),flush=True)

if __name__=='__main__':main()
