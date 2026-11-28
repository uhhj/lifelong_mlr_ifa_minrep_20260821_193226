"""Calibrate the alternative wrist measurement on successful training episodes only."""
from basis import *
from wrist_lines import visible_face
from b1_alignment_gate import wrist_centered
def measure():
    from assess import reviewed
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    refs=reviewed(work,prior)
    sets=[[r for _,r in rows('train')],[dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in refs if r['train']]]
    env=read(edge/'view_geometry.json')['close_envelopes']
    result=[]
    for group,recs in enumerate(sets):
        widths=[]
        found=0
        for rec in recs:
            c=rec['close']
            if c is None:continue
            w=rp.at(rec['rows'][c],'feedback')[6]
            box=visible_face(rp.imaging(rec,c,'wrist'))
            found+=int(box is not None)
            if wrist_centered(box,w):widths.append(float(box[2]))
        minimum=float(np.percentile(widths,10)) if len(widths)>=8 else env[group]['min_face_width']
        result.append(dict(measured=found,centered=len(widths),min_width=minimum))
    dump('widths.json',result)
    print(result,flush=True)
if __name__=='__main__':
    sys.path.insert(0,str(work/'offline_diagnostics/b1_vision_contrasts_20261124'))
    cv2.setNumThreads(1);measure()
