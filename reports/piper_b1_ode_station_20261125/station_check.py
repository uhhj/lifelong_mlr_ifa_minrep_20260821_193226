"""Exercise the registered model through the existing console using archived observations."""
import json
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import torch

from station_install import DIAG, KEY


def reset(model):
    model.history.clear();model.chunk=None;model.last_result=None
    model.last_stamp=0;model.execution=None


def query(q,stamp,anchor=None,previous=0):
    active=anchor is not None
    return dict(q=q.tolist(),execution=int(active),mode_at_query='execute' if active else 'preview',
                previous_sent=anchor.tolist() if active else None,previous_stamp=previous,feedback_ns={})


def feed(model,rec,other):
    reset(model)
    for q,stamp,head,wrist in zip(rec['q'],rec['stamps'],other['overhead'],rec['wrist']):
        target=model.predict_recorded(model.prepare_views((head,wrist)),query(q,stamp,rec['anchor'],model.last_stamp),stamp)
    assert model.trace['ready'] and model.trace['offset']==0
    np.testing.assert_array_equal(target,model.chunk[0])
    return model.chunk.copy(),dict(model.trace['alignment'])


def examples(work,prior):
    import sys
    sys.path.insert(0,str(prior))
    from assess import reviewed
    from annotate import images
    data=json.loads((prior/'final.json').read_bytes())['geometry']
    out={}
    for r in reviewed(work,prior):
        ep=r['episode']
        if ep not in data['test_episodes']:continue
        t=r['close']-18
        start=r['first']+t-7
        lines=(r['folder']/'alignment.jsonl').read_bytes().splitlines()[start:start+8]
        item=dict(q=r['q'][t-7:t+1],anchor=r['cmd'][t-1],
                  stamps=[json.loads(s)['sample_monotonic_ns'] for s in lines],uv=r['uv'])
        for view in ('overhead','wrist'):
            item[view]=[cv2.imread(str(p)) for p in images(r['folder'],view)[start:start+8]]
        out[ep]=item
    return data,out


def protocol(model,item):
    base,details=feed(model,item,item)
    assert details['applied']
    t=model.last_stamp+33_333_333;q=item['q'][-1]
    pix=model.prepare_views((item['overhead'][-1],item['wrist'][-1]))
    try:model.predict_recorded(pix,query(q,t,base[0],model.last_stamp-1),t)
    except ValueError as e:assert '确认' in str(e)
    else:raise AssertionError('unconfirmed command accepted')
    feed(model,item,item)
    closed=q.copy();closed[6]=45.
    model.predict_recorded(pix,query(closed,t),t)
    assert not model.trace['alignment']['applied'] and model.closed
    np.testing.assert_array_equal(model.chunk,model.base_plan)
    model.chunk=None
    model.predict_recorded(pix,query(q,t+33_333_333),t+33_333_333)
    assert model.closed and not model.trace['alignment']['applied']
    fresh=t+160_000_000
    model.predict_recorded(pix,query(q,fresh),fresh)
    assert len(model.history)==len(model.raw)==1 and not model.trace['ready']
    return dict(acknowledgement=True,closed_and_reopened_bypass=True,gap_clears_both_histories=True)


def timed(work,stage,model,item):
    from capture import FrameArchive
    reset(model)
    times=[];references=[];ncorrect=0
    with tempfile.TemporaryDirectory(dir=stage) as tmp:
        folder=Path(tmp)/'replay'
        with FrameArchive(folder) as archive:
            for i in range(85):
                j=min(i,7);stamp=1_000_000_000+i*33_333_333
                images=[item[v][j] for v in ('overhead','wrist')]
                pair=[dict(image=f,host_monotonic_ns=stamp,read_started_monotonic_ns=stamp,sequence=i) for f in images]
                # Includes a replan on every eligible tick, as happens with clipped commands.
                model.chunk=None
                t=time.perf_counter_ns()
                pred,refs=archive.predict(model,pair,model.prepare_views(images),query(item['q'][j],stamp,item['anchor'],model.last_stamp),stamp)
                dt=(time.perf_counter_ns()-t)/1e6
                if i>=8:times.append(dt);ncorrect+=int(model.trace['alignment']['applied'])
                if i in (0,8,84):references.append((refs,images))
        for ref,imgs in references:
            assert ref['model']==ref['scene']
            np.testing.assert_array_equal(cv2.imread(str(folder/ref['scene'])),np.concatenate(imgs,axis=1))
        rows=[json.loads(r) for r in (folder/'inputs.jsonl').read_text().splitlines()]
        assert len(rows)==85 and all(r['stamp']==r['history'][-1] for r in rows)
        assert all('alignment' in r['plan'] for r in rows[7:])
    assert ncorrect==len(times)
    return dict(replans=len(times),corrected=ncorrect,archive='lossless raw pair',median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times,95)),max_ms=max(times),over_100ms=sum(x>=100 for x in times))


def field_guard_replay(work,runner,model):
    """Replay archived execution inputs and the two actual tracking faults without a CAN endpoint."""
    result=[]
    for name in ('20261123_234228_1446439_B1-ode50-20261123','20261123_234953_1448999_B1-ode50-20261123'):
        directory=work/'policy_runs'/name
        inputs=[json.loads(s) for s in (directory/'inputs.jsonl').read_bytes().splitlines()]
        run=[json.loads(s) for s in directory.with_suffix('.jsonl').read_bytes().splitlines()]
        sent={r['stamp']:r for r in run if r.get('sent') is not None}
        fault=next(r for r in run if r['event']=='target_guard_fault')
        by_stamp={r['stamp']:r for r in inputs}
        active=[r for r in inputs if r['mode_at_query']=='execute']
        # Rebuild the first active observation's real preview history from disk.
        seed=[by_stamp[s] for s in active[0]['history'][:-1]]
        reset(model)
        last_sent=None;last_stamp=0;origin=0;counts={};last_pred=None
        for r in seed+active:
            stamp=r['stamp']
            im=cv2.imread(str(directory/r['images']['scene']))
            pix=model.prepare_views((im[:,:640],im[:,640:]))
            is_active=r['mode_at_query']=='execute'
            if is_active and last_sent is None:
                origin=stamp
                last_sent=np.array(r['plan']['anchor'])
            q=np.asarray(r['q'])
            request=query(q,stamp,last_sent if is_active else None,last_stamp)
            if is_active:request['execution']=origin
            prediction=model.predict_recorded(pix,request,stamp)
            if is_active and model.trace['ready']:
                phase=model.trace['alignment']['phase']
                counts[phase]=counts.get(phase,0)+1
                last_pred=prediction
            if stamp in sent:
                last_stamp=stamp
                last_sent=np.asarray(sent[stamp]['sent'])
        assert last_pred is not None and counts.get('approach',0)>0,(name,counts)
        try:
            runner.bounded_target(last_pred,fault['measured_at_send'],fault['previous_sent'],1/30,b3_tracking=True)
        except runner.TargetGuardFault as e:
            axes=[v['axis'] for v in e.violations]
            assert '夹爪' in axes
        else:raise AssertionError('Original field tracking fault was not rejected')
        result.append(dict(session=name,active_predictions=sum(counts.values()),phase_counts=counts,rejected_axes=axes))
    return result


def replay(work,stage,runner,p,full):
    cv2.setNumThreads(1);torch.set_num_threads(4)
    model=runner.load_policy(p)
    assert runner.FRESH_NS==100_000_000 and runner.SAFETY_NS==500_000_000
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    geometry,records=examples(work,prior)
    historical=json.loads((work/DIAG/'causal_appearance/outcomes.json').read_bytes())
    nat={r['episode']:r for r in historical['natural'] if r['mode']=='settle/feedback'}
    swaps={(r['recipient'],r['donor']):r for r in historical['directions'] if r['mode']=='settle/feedback'}
    matrix=np.asarray(geometry['table_map']);pairs=geometry['pairs']
    confirmed=[];positions={};flags={};best=None
    for ep,r in records.items():
        result,details=feed(model,r,r)
        ref=model.base_plan
        if len(confirmed)<3:
            with torch.inference_mode():
                rgb=torch.as_tensor(np.stack([p for p,_ in model.history]),device='cuda').permute(1,0,4,2,3).contiguous().float()/255
                obs=dict(overhead=rgb[:1],wrist=rgb[1:],agent_pos=torch.as_tensor(r['q'][None],device='cuda'))
                anchor=torch.as_tensor(r['anchor'][None],device='cuda')
                direct=model.model.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].cpu().numpy()
            np.testing.assert_allclose(ref[:,:6],direct[:,:6],atol=1e-5,rtol=0)
            np.testing.assert_allclose(ref[:,6],direct[:,6],atol=.001,rtol=0)
        if details['applied']:
            expected,_=model.aligner.predict(r['overhead'],r['q'][-1],ref,qhistory=r['q'],wrist=r['wrist'][-1])
            if best is None:best=r
        else:expected=ref
        np.testing.assert_allclose(result,expected,atol=1e-5,rtol=0)
        np.testing.assert_array_equal(result[:,6],ref[:,6])
        positions[ep]=model.aligner.fk(result[-1]);flags[ep]=details['applied']
        confirmed.append(dict(episode=ep,shape=list(result.shape),applied=details['applied'],
            phase=details['phase'],max_abs=float(np.abs(result-expected).max())))
        if len(confirmed)==3 and not full:break
    if not full:return dict(windows=confirmed)
    assert len(confirmed)==33 and best is not None
    endpoint=[];matched=[];directions=[];toward=0
    for a,b in pairs:
        group=[]
        for ep,other in ((a,b),(b,a)):
            r=records[ep];donor=records[other]
            joint,d=feed(model,r,donor)
            final=model.aligner.fk(joint[-1])
            goal=np.r_[donor['uv'],1.]@matrix
            axis=goal-np.r_[r['uv'],1.]@matrix
            signed=float((final-positions[ep])@axis/np.linalg.norm(axis))
            group.append(signed)
            toward+=int(np.linalg.norm(goal-final)<np.linalg.norm(goal-model.aligner.fk(r['q'][-1])))
            if d['applied'] and flags[ep]:
                matched.append(abs(signed-swaps[ep,other]['signed_mm']))
            directions.append(dict(recipient=ep,donor=other,signed_mm=signed,applied=d['applied'],phase=d['phase']))
        endpoint.append(np.mean(group))
    assert matched and max(matched)<.05,('reference mismatch',max(matched))
    checks=dict(direction_positive=int(np.sum(np.asarray(endpoint)>0)),over_1mm=int(np.sum(np.asarray(endpoint)>1)),
        below_minus_1mm=int(np.sum(np.asarray(endpoint)<-1)),median_signed_mm=float(np.median(endpoint)),
        matched_directions=len(matched),max_difference_to_offline_mm=max(matched),individual_closer=toward)
    print('held-out profile',checks,flush=True)
    safety=protocol(model,best)
    field=field_guard_replay(work,runner,model)
    timing=timed(work,stage,model,best)
    print('archive and inference timing',timing,flush=True)
    assert checks['direction_positive']>=14 and checks['over_1mm']>=12 and checks['below_minus_1mm']<=1 and toward>=24
    assert timing['p95_ms']<1000/30 and timing['over_100ms']==0,('latency',timing)
    return dict(profile=KEY,sha256=p['sha256'],windows=confirmed,archive_anchor_count=len(nat),directions=directions,
                pair_checks=checks,timing=timing,protocol=safety,field=field,accepted=True)
