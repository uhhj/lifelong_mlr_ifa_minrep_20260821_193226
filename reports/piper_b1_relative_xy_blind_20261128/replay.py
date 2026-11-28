"""Replay matched B1 predictions from synchronized files without device interfaces."""
from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json, sys
import cv2
import numpy as np
import torch

here=Path(__file__).resolve().parent
work,old=map(Path,sys.argv[1:3])
sys.path.insert(0,str(old))
def read(p):return json.loads(p.read_bytes())
def dump(name,data):(here/name).write_text(json.dumps(data,allow_nan=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(l) for l in p.read_bytes().splitlines()]
def imaging(rec,j,role):
    stamp=rec['rows'][j]['sample_monotonic_ns']
    path=rec['folder']/'episode0/camera/color'/role/f'{stamp/1e9:.6f}.jpg'
    result=cv2.imread(str(path))
    if result is None or result.shape!=(480,640,3):raise ValueError(f'Missing image {path.name}')
    return result
def at(r,k):return r[k]['joints_rad']+[r[k]['gripper_mm']]
def valid(row):
    return all(row.get(k) and row[k].get('joints_rad') is not None and row[k].get('gripper_mm') is not None for k in ('target','feedback'))
def load_record(folder,first=0):
    seq=lines(folder/'alignment.jsonl')[first:]
    return dict(folder=folder,rows=seq)
def blind_records():
    frozen=read(here/'frozen.json')
    recs={}
    for ep in frozen['blind']:
        rec=load_record(work/'batches/B1_20261123/failed'/ep)
        seq=rec['rows']
        cross=next((i for i in range(1,len(seq)) if valid(seq[i-1]) and valid(seq[i])
                    and at(seq[i-1],'target')[6]>55>=at(seq[i],'target')[6]),None)
        rec['close']=cross
        recs[ep]=rec
    return recs
def board():
    tiles=[]
    for ep,r in blind_records().items():
        j=next(i for i,v in enumerate(r['rows']) if valid(v))
        im=imaging(r,j,'overhead')
        # Known raw sensor coordinates, no proposals or inference overlays.
        tiles.append(cv2.copyMakeBorder(im[242:452,104:424],22,0,0,0,cv2.BORDER_CONSTANT))
        cv2.putText(tiles[-1],ep,(3,16),0,.48,(255,255,255),1)
    tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(here/'blind_first.jpg'),np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,8,2)]))
    panels=[]
    for ep,r in blind_records().items():
        c=r['close'];n=len(r['rows'])
        times=sorted(set([0,max(0,(c or n//2)-27),max(0,(c or n//2)-18),max(0,(c or n//2)-7),min(n-1,c or n//2)]))
        ims=[]
        for j in times:
            im=imaging(r,j,'overhead')[270:425,175:320].copy()
            for x in range(180,320,20):
                for y in range(280,425,20):cv2.circle(im,(x-175,y-270),1,(255,128,0),-1)
            tile=cv2.copyMakeBorder(im,20,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,f"{ep[9:15]} {j}/{c}",(1,15),0,.4,(255,255,255),1);ims.append(tile)
        while len(ims)<5:ims.append(np.zeros_like(ims[0]))
        panels.append(np.hstack(ims))
    cv2.imwrite(str(here/'blind_grid.jpg'),np.vstack(panels))

def models():
    def make(n,f,p):
        spec=importlib.util.spec_from_file_location(n,f);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod.GroundedODE(p)
    p=read(old/'profiles.json')['B1-ode50-aligned-20261125']
    c=read(here/'candidate.json')
    frozen=read(here/'frozen.json')
    assert [p['sha256'],c['sha256']]==[frozen['baseline'],frozen['candidate']]
    live=work/'piper_test_console_pika_b1_full_v1'
    assert sha(old/'ode_alignment.py')==sha(live/'ode_wrist_alignment.py')
    for n in ('bus.py','b1_placement.py','b1_gripper.py'):
        assert sha(old/n)==sha(live/n),n
    sys.path.insert(0,str(live))
    return [make('frozen_b1',old/'ode_alignment.py',p),make('relative_b1',here/'ode_relative.py',c)]

def reset(m):
    m.history.clear();m.raw.clear();m.heat.clear();m.features.clear()
    m.execution=None;m.last_stamp=0;m.chunk=None;m.last_result=None
    m.mode=m.plan_mode=None

def endpoint(m,r,donor):
    reset(m)
    j=r['close']-18;k=donor['close']-18
    # Restore the same actual recipient history. Swap only the final eight overhead frames.
    for i in range(max(0,j-15),j+1):
        row=r['rows'][i]
        if not valid(row):
            reset(m)
            continue
        t=row['sample_monotonic_ns'];q=at(row,'feedback')
        over=imaging(donor,k+i-j,'overhead') if i>=j-7 else imaging(r,i,'overhead')
        wrist=imaging(r,i,'wrist')
        prev=at(r['rows'][i-1],'target') if i and valid(r['rows'][i-1]) else q
        if i==j:m.chunk=None
        query=dict(q=q,execution=1,mode_at_query='execute',previous_sent=prev,previous_stamp=m.last_stamp,gripper_phase='approach')
        m.predict_recorded(m.prepare_views([over,wrist]),query,t)
    assert m.trace['ready'] and m.trace['offset']==0
    align=m.aligner
    def pose(plan):
        return align.xy.pose(np.asarray(plan)[:,:6].astype(float),align.par['tip'])
    xyz,rot=pose(m.chunk);base,base_rot=pose(m.base_plan)
    assert np.max(abs(xyz[:,2]-base[:,2]))<.001 and np.max(abs(rot-base_rot))<1e-5
    assert np.array_equal(m.chunk[:,6],m.base_plan[:,6])
    return xyz[-1,:2],align.fk(q)

def eligible(r):
    if r['close'] is None:return False
    j=r['close']-18
    if j<7:return False
    s=r['rows'][j-7:j+1]
    return bool(all(valid(row) for row in s) and max(np.diff([row['sample_monotonic_ns'] for row in s]))<=70e6)

def close_event(m,r):
    from runner import GraspIntent
    from b1_placement import B1Placement
    gate=B1Placement();intent=GraspIntent();reset(m)
    end=min(len(r['rows']),r['close']+9) if r['close'] is not None else len(r['rows'])
    for i,row in enumerate(r['rows'][:end]):
        if not valid(row) or (m.last_stamp and row['sample_monotonic_ns']-m.last_stamp>70_000_000):
            gate=B1Placement();intent=GraspIntent();reset(m)
            if not valid(row):continue
        stamp=row['sample_monotonic_ns'];q=at(row,'feedback')
        prev=at(r['rows'][i-1],'target') if i and valid(r['rows'][i-1]) else q
        views=[imaging(r,i,role) for role in ('overhead','wrist')]
        query=dict(q=q,execution=1,mode_at_query='execute',previous_sent=prev,previous_stamp=m.last_stamp,gripper_phase='approach')
        m.predict_recorded(m.prepare_views(views),query,stamp)
        if not m.trace['ready']:continue
        intent.observe(stamp,q[6],stamp,m.intent_widths)
        tip=m.aligner.flange(np.rad2deg(q[:6]));xyz=tip[:3,3]+tip[:3,:3]@m.aligner.par['tip']
        goal=gate.next_goal(stamp,intent.fired,xyz,q[6],vision=m.visual)
        if goal==35:
            return dict(frame=i,relative_to_expert=i-r['close'] if r['close'] is not None else None)
    return None

def select_blind_pairs(recs,goals,fk):
    import networkx as nx
    graph=nx.Graph()
    for i,a in enumerate(sorted(recs)):
        for b in sorted(recs)[:i]:
            if not eligible(recs[a]) or not eligible(recs[b]):continue
            qa,qb=[np.array(at(recs[e]['rows'][recs[e]['close']-18],'feedback')) for e in (a,b)]
            d=np.linalg.norm(fk(qa)-fk(qb));angle=np.rad2deg(abs(qa[:6]-qb[:6]));width=abs(qa[6]-qb[6])
            if np.linalg.norm(goals[a]-goals[b])>=20 and d<=55 and angle.max()<=25 and width<=20:
                graph.add_edge(a,b,weight=-float(d/55+np.linalg.norm(angle)/25+width/20))
    return sorted(sorted(p) for p in nx.max_weight_matching(graph,maxcardinality=True))

def archive_records(model):
    manifest=read(here/'old_manifest.json')
    allowed={e['episode_id']:e for e in manifest['unselected']}
    assert len(allowed)==20 and set(allowed).isdisjoint(manifest['selected'])
    used=set()
    for folder in (work/'offline_diagnostics').iterdir():
        if not folder.is_dir() or folder.name.startswith('b1-relative.'):continue
        for pat in ('*.json','*/*.json','*.py','*.md'):
            for path in folder.glob(pat):
                if path.is_file() and path.stat().st_size<32*1024**2:
                    blob=path.read_bytes()
                    used.update(ep for ep in allowed if ep.encode() in blob)
    recs={};prior=[];goals={}
    for ep in [*manifest['selected'],*sorted(set(allowed)-used)]:
        folder=work/'recordings'/ep
        if ep in allowed:assert sha(folder/'alignment.jsonl')==allowed[ep]['alignment_sha256']
        rec=load_record(folder)
        seq=rec['rows']
        c=next((i for i in range(1,len(seq)) if valid(seq[i-1]) and valid(seq[i]) and at(seq[i-1],'target')[6]>55>=at(seq[i],'target')[6]),None)
        contact=next((i for i in range(max(0,(c or 0)-6),len(seq)-2) if all(valid(v) and at(v,'feedback')[6]<=49 for v in seq[i:i+3])),None) if c is not None else None
        dest=model.aligner.fk(at(seq[contact],'feedback')) if contact is not None else None
        if ep in allowed:
            rec['close']=c
            recs[ep]=rec
            if dest is not None:goals[ep]=dest
        elif dest is not None:prior.append(dest)
    return recs,goals,prior,sorted(used)

def evaluate(group):
    cv2.setNumThreads(1);torch.set_num_threads(2)
    two=models()
    from assess import reviewed
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    geo=read(prior/'final.json')['geometry'];gt=np.asarray(geo['table_map'])
    refs=reviewed(work,prior)
    train=np.array([r['uv'] for r in refs if r['train'] and r['uv'] is not None])
    midpoint=np.median(train,axis=0)
    if group=='regression':
        pairs=geo['pairs'];ids={e for pair in pairs for e in pair}
        recs={r['episode']:dict(**load_record(r['folder'],r['first']),uv=r['uv'],close=r['close']) for r in refs if r['episode'] in ids}
        assert len(pairs)==16 and len(recs)==32
        goals={e:np.r_[r['uv'],1.]@gt for e,r in recs.items()}
    elif group=='blind':
        assert (here/'regression.json').exists()
        recs=blind_records()
        labels=read(here/'blind_labels.json')
        for e,r in recs.items():r['uv']=labels[e]
        goals={e:np.r_[r['uv'],1.]@gt for e,r in recs.items() if r['uv'] is not None}
        pairs=select_blind_pairs({e:recs[e] for e in goals},goals,two[0].aligner.fk)
    elif group=='archive':
        recs,goals,older,excluded=archive_records(two[0])
        for e,r in recs.items():r['uv']=np.linalg.solve(gt[:2].T,goals[e]-gt[2]).tolist() if e in goals else None
        pairs=select_blind_pairs({e:recs[e] for e in goals},goals,two[0].aligner.fk)
    else:raise ValueError(group)
    # Include all 84 previously used positions and identified recent field centers.
    known=[np.r_[r['uv'],1.]@gt for r in refs if r['uv'] is not None]
    for f in sorted((work/'policy_runs').glob('*B1*.jsonl')):
        with f.open() as stream:
            for line in stream:
                row=json.loads(line);v=row.get('grasp_alignment') or {}
                if row.get('mode')=='execute' and row.get('sent') and v.get('block_uv') is not None:
                    known.append(np.r_[v['block_uv'],1.]@gt);break
    out=dict(group=group,median_uv=midpoint.tolist(),pairs=pairs,episodes={},directions=[])
    if group=='archive':
        known.extend(older)
        out.update(prior_old_contacts=len(older),excluded_prior_mentions=excluded)
    ends={}
    for i,(ep,r) in enumerate(recs.items()):
        uv=r['uv']
        region=None if uv is None else ('左' if uv[0]<midpoint[0] else '右')+('上' if uv[1]<midpoint[1] else '下')
        record=dict(region=region,eligible=eligible(r),close_frame=r['close'],alignment_sha256=sha(r['folder']/'alignment.jsonl'),
                    nearest_prior_mm=None if uv is None else round(float(np.min(np.linalg.norm(np.array(known)-goals[ep],axis=1))),2))
        if record['eligible'] and ep in goals:
            ends[ep]=[endpoint(m,r,r) for m in two]
            record['natural_mm']=[round(float(np.linalg.norm(start-goals[ep])-np.linalg.norm(end-goals[ep])),4) for end,start in ends[ep]]
        record['close_events']=[close_event(m,r) for m in two]
        out['episodes'][ep]=record
        dump(group+'_partial.json',out)
        if (i+1)%5==0:print(group,'natural',i+1,flush=True)
    for a,b in pairs:
        for e,donor in ((a,b),(b,a)):
            axis=goals[donor]-goals[e];axis=axis/np.linalg.norm(axis)
            responses=[];advances=[]
            for idx,m in enumerate(two):
                end,start=endpoint(m,recs[e],recs[donor])
                responses.append(round(float((end-ends[e][idx][0])@axis),5))
                advances.append(round(float(np.linalg.norm(start-goals[donor])-np.linalg.norm(end-goals[donor])),5))
            out['directions'].append(dict(recipient=e,donor=donor,region=out['episodes'][donor]['region'],
                                          correction_mm=responses,advance_mm=advances))
    dump(group+'.json',out)
    tiles=[];mapping={}
    for ep,rec in out['episodes'].items():
        for ev in rec['close_events']:
            if not ev:continue
            token=hashlib.sha256((ep+str(ev['frame'])).encode()).hexdigest()[:8]
            if token in mapping:continue
            mapping[token]=[ep,ev['frame']]
    for token,(ep,i) in sorted(mapping.items()):
        im=imaging(recs[ep],i,'wrist')
        tile=cv2.resize(im,(320,240))
        tile=cv2.copyMakeBorder(tile,20,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(tile,token,(5,15),0,.45,(255,255,255),1);tiles.append(tile)
    if tiles:
        if len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(here/(group+'_events.jpg')),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    dump(group+'_event_key.json',mapping)
    print('finished',group,len(recs),len(pairs),len(mapping),flush=True)

if __name__=='__main__':
    if sys.argv[3]=='board':board()
    else:evaluate(sys.argv[3])
