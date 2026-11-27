"""Extract archived B1 contact references and initial synchronized views."""
import json,sys
from pathlib import Path
import numpy as np
import cv2


def references(work):
    selection=json.loads((work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json').read_bytes())
    ranking={r['id']:r for r in selection['ranking']}
    ref=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    labels=json.loads((ref/'points.json').read_bytes())
    for k,v in json.loads((ref/'annotations.json').read_bytes()).items():labels[int(k)].update(v)
    return [(lab,ranking[lab['episode']]) for lab in labels if lab['episode'] in ranking and lab['close'] is not None]


def main(work):
    profile=json.loads((Path(__file__).parent/'profiles.json').read_bytes())['B1-ode50-aligned-20261125']
    bundle=profile['alignment']
    sys.path.insert(0,str((work/bundle['algorithm']).parent))
    import grounded
    grounded.setup(work)
    from fit_gripper import flange
    weights=work/bundle['weights']
    with np.load(weights/'private_policy.npz') as data:tip=data['tip']
    dest=Path(__file__).resolve().parent
    if len(sys.argv)>2 and sys.argv[2]=='axes':
        from audit_axes import audit
        audit(tip,flange)
        return
    if len(sys.argv)>2 and sys.argv[2]=='camera':
        from camera_fit import fit
        fit(work)
        return
    samples=[];poses=[];ids=[];arm=[];images=[]
    for lab,e in references(work):
        folder=work/'batches/B1_20261123'/e['source']
        rows=[json.loads(l) for l in (folder/'alignment.jsonl').read_bytes().splitlines()][e['first']:]
        c=lab['close']
        if c>=len(rows):continue
        contact=lab.get('contact')
        if contact is None:contact=c
        index=[0,max(0,c-32),max(0,c-20),max(0,c-12),max(0,c-8),max(0,c-4),c,contact]
        seq=[];qseq=[]
        for j in index:
            q=rows[j]['feedback']['joints_rad']
            stamp=rows[j]['sample_monotonic_ns']
            pair=[cv2.imread(str(folder/'episode0/camera/color'/role/f'{stamp/1e9:.6f}.jpg')) for role in ('overhead','wrist')]
            seq.append([cv2.resize(im,(320,240),interpolation=cv2.INTER_AREA) for im in pair])
            qseq.append(q)
        qs=np.asarray(qseq);T=flange(np.rad2deg(qs));xyz=T[:,:3,3]+T[:,:3,:3]@tip
        samples.append(dict(episode=lab['episode'],train=lab['train'],close=c,contact=contact,indices=index))
        ids.append(lab['episode']);poses.append(xyz);arm.append(qs)
        images.append(seq)
    np.savez_compressed(dest/'private_refs.npz',xy=poses,q=arm,im=np.asarray(images),ids=ids)
    (dest/'refs.json').write_text(json.dumps(samples))
    print('References',len(samples),'example label keys',labels_desc(lab),flush=True)


def labels_desc(lab):
    return list(lab)


if __name__=='__main__':main(Path(sys.argv[1]))
