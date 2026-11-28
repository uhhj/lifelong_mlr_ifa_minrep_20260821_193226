"""Read archived episodes and frozen prediction profiles without constructing device interfaces."""
import sys,json,hashlib,math,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
work,old,prev,edge,frozen=map(Path,sys.argv[1:6])
dest=Path(__file__).resolve().parent
sys.path.insert(0,str(frozen))
from basis import rp,rows,read,model
sys.path[:0]=[str(work/'piper_test_console_pika_b1_full_v1'),str(work/'offline_diagnostics/b1_vision_contrasts_20261124')]
from phases import classify
def dump(name,v):(dest/name).write_text(json.dumps(v,allow_nan=False,separators=(',',':')))
def subjects(group):
    meta=read(frozen/(group+'.json'))['episodes']
    if group=='archive':
        points={r['ep']:np.array(r['label']) for r in read(edge/'diag.json')['test']}
        M=np.array(read(edge/'view_geometry.json')['archive_matrix'])
        records=dict(rows('test'))
    else:
        from assess import reviewed
        prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
        refs=[r for r in reviewed(work,prior) if r['episode'] in meta]
        M=np.array(read(prior/'final.json')['geometry']['table_map'])
        records={r['episode']:dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in refs}
        points={r['episode']:r['uv'] for r in refs}
    assert set(meta)==set(records)
    for e,r in records.items():
        assert r['close']==meta[e]['close_frame'] and rp.sha(r['folder']/'alignment.jsonl')==meta[e]['alignment_sha256']
        r['visual_goal']=np.r_[points[e],1]@M
        yield e,r

def models():
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    cand=read(frozen/'candidate.json')
    assert cand['sha256']=='89cd61aac0bcfa63d9bd8676385b0ad1c0206f7c1aaaf126db031a6f40acfd4b'
    local=work/'piper_test_console_pika_b1_full_v1'
    field_profile=read(old/'profiles.json')['B1-ode50-aligned-20261125']
    assert field_profile['alignment']['base_sha256']==cand['alignment']['base_sha256']
    assert rp.sha(old/'ode_alignment.py')==rp.sha(local/'ode_wrist_alignment.py')
    assert rp.sha(old/'bus.py')==rp.sha(local/'bus.py')
    def load_field():
        spec=importlib.util.spec_from_file_location('field_b1_diag',old/'ode_alignment.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod.GroundedODE(field_profile)
    return {'candidate':model(frozen,'ode_tracked.py'),'field':load_field()}

def xyz(m,q):
    q=np.asarray(q,dtype=float)
    T=m.aligner.flange(np.rad2deg(q[...,:6]))
    return T[...,:3,3]+T[...,:3,:3]@m.aligner.par['tip']
def vec(a):return np.asarray(a).round(5).tolist()
def recorded(rec,m):
    seq=rec['rows'];c=rec['close']
    good=np.array([rp.valid(x) for x in seq]);stamps=np.array([x['sample_monotonic_ns'] for x in seq],np.int64)
    fb=np.array([rp.at(x,'feedback') if g else [0]*7 for x,g in zip(seq,good)])
    cmd=np.array([rp.at(x,'target') if g else [0]*7 for x,g in zip(seq,good)])
    x=xyz(m,fb);cmdx=xyz(m,cmd)
    contact=next(i for i in range(max(0,c-6),len(seq)-2) if good[i:i+3].all() and np.all(fb[i:i+3,6]<=49))
    goal=x[contact,:2]
    return good,stamps,fb,cmd,x,cmdx,contact,goal
def visual_step(m,views,stamp,q,prevq,phase='approach'):
    return m.predict_recorded(m.prepare_views(views),dict(q=q,execution=1,mode_at_query='execute',previous_sent=prevq,previous_stamp=m.last_stamp,gripper_phase=phase),int(stamp))
def fresh(m,anchor):
    if m.trace['offset']==0:return m.base_plan,m.chunk,m.details.copy()
    saved=(m.base_plan,m.plan_mode,m.details,m.guiding)
    try:
        plan=m.plan(np.asarray(anchor,np.float32))
        return m.base_plan,plan,m.details.copy()
    finally:m.base_plan,m.plan_mode,m.details,m.guiding=saved
