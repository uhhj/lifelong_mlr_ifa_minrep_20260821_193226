"""Recompute the archived XY plan and compare it to recorded pred/command/feedback."""
from pathlib import Path
import collections
import hashlib
import json
import sys


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(work, name):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    folder=work/'policy_runs'/name
    path=folder.with_suffix('.jsonl')
    rows=read(path)
    refs=rows[0]['policy']['alignment']
    script=work/refs['algorithm']
    weights=work/refs['weights']
    for artifact in (script,weights/'private_policy.npz',weights/'private_settle.npz'):
        assert digest(artifact)==refs['files'][str(artifact.relative_to(work))]
    sys.path.insert(0,str(script.parent))
    import grounded as alg
    alg.setup(work)
    from fit_gripper import flange
    with np.load(weights/'private_policy.npz') as archive: par=dict(archive)
    with np.load(weights/'private_settle.npz') as archive: settle=dict(archive)
    def tip(q):
        pose=flange(np.rad2deg(np.asarray(q,dtype=float)[:6]))
        return (pose[:3,3]+pose[:3,:3]@par['tip'])[:2]
    def norm(v):return float(np.linalg.norm(v))
    def stats(values):return dict(min=round(min(values),4),median=round(float(np.median(values)),4),max=round(max(values),4))
    inputs={r['stamp']:r for r in read(folder/'inputs.jsonl')}
    sent=[r for r in rows if r.get('sent') and r.get('mode')=='execute']
    t0=next(r['mono_ns'] for r in rows if r['event']=='gripper_stable')
    trace=next(r for r in rows if r['event']=='pika_stop_trace')
    plans={}
    geometry=[]
    tracking=[]
    for r in sent:
        sample=inputs[r['stamp']]
        plan=sample['plan']
        when=plan['chunk_stamp']
        if when not in plans:
            head=inputs[when]
            h=np.asarray([inputs[t]['q'] for t in head['history']],dtype=np.float32)
            assert h.shape==(8,7)
            frame=cv2.imread(str(folder/head['images']['model']))
            assert frame.shape==(480,1280,3) and head['plan']['alignment']['applied']
            wrist=frame[:,640:].copy()
            uv=np.array(head['plan']['alignment']['block_uv'])
            raw=np.r_[uv,1.]@par['matrix']
            x,w=alg.context_features(h,wrist,flange,par['tip'])
            pos=x[:2]
            inferred=settle['b']+((x-settle['xm'])/settle['xsd']/np.sqrt(len(x)))@settle['wx']+((w-settle['wm'])/settle['wsd'])@settle['ww']
            cue=raw-pos-inferred[2:]
            z0,ramp,quiet,span=settle['guard']
            height_term=float(np.clip((x[2]-settle['contact_z']-z0)/ramp,0,1))
            cue_term=float(np.clip((norm(cue)-quiet)/span,0,1))
            weight=float(alg.close_weight(x[2],settle['contact_z'],cue,settle['guard']))
            assert abs(weight-max(height_term,cue_term))<1e-6
            pre_weight={k:v for k,v in settle.items() if k!='guard'}
            unattenuated=alg.settle_goal(uv,h,wrist,par,pre_weight,flange)
            goal=alg.settle_goal(uv,h,wrist,par,settle,flange)
            assert np.allclose(goal-pos,weight*(unattenuated-pos),atol=1e-6)
            endpoint=alg.move(par,pos,goal,'feedback','heatmap')
            alternative=alg.move(par,pos,unattenuated,'feedback','heatmap')-pos
            direction=(raw-pos)/max(1e-9,norm(raw-pos))
            plans[when]=dict(seconds=(when-t0)/1e9,pos=pos,endpoint=endpoint,
                             raw_mm=norm(raw-pos),weight=weight,filtered_mm=norm(goal-pos),
                             height_weight=height_term,cue_weight=cue_term,
                             before_weight_mm=norm(unattenuated-pos),eight_step_mm=norm(endpoint-pos),
                             hypothetical_eight_step_mm=norm(alternative),
                             along_overhead_mm=float((endpoint-pos)@direction),
                             hypothetical_along_overhead_mm=float(alternative@direction))
        a=plans[when]
        expected=a['pos']+(plan['offset']+1)/8*(a['endpoint']-a['pos'])
        geometry.append(norm(tip(r['p'])-expected))
        tracking.append(dict(t=(r['stamp']-t0)/1e9,
                             pred_sent=norm(tip(r['p'])-tip(r['sent'])),
                             sent_fb=norm(tip(r['sent'])-tip(r['measured_at_send'])),
                             offsets=plan['offset']))
    assert max(geometry)<.01,max(geometry)
    slow=[p for p in plans.values() if p['seconds']>=4.]
    ticks=[p for p in tracking if p['t']>=4.]
    last4=next(r for r in sent if r['stamp']>=t0+4_000_000_000)
    phase=collections.Counter(r.get('grip_plan_phase') for r in sent)
    aligned=[r['grasp_alignment'] for r in sent]
    boxed=[r for r in sent if r['grasp_alignment']['wrist_box'] is not None]
    def failure(r):
        left,top,width,height=r['grasp_alignment']['wrist_box']
        clearance=max(0.,(r['q'][6]-35.)/2-2.)*width/35.
        return dict(horizontal=abs(left+width/2-355)>min(42.,clearance),depth=not(276<=top<=372 and top+height>=405))
    return dict(session=name,source_sha256=digest(path),inputs_sha256=digest(folder/'inputs.jsonl'),
                verified_plan_frames=len(geometry),max_xy_reconstruction_error_mm=max(geometry),
                model_bundle=rows[0]['policy']['sha256'],execution_duration_setting_s=rows[0]['policy']['duration'],
                stop_after_s=(trace['stop_requested_ns']-t0)/1e9,end_detail=rows[-1]['detail'],width_targets_mm=[r['pika_target_mm'] for r in sent if r['pika_target_mm'] is not None],
                width_feedback_mm=stats([r['measured_at_send'][6] for r in sent]),
                phases=phase,close_intent=sum(r['close_intent'] for r in sent),
                passed=sum(v['aligned'] for v in aligned),overhead_within_20=sum(v['distance_mm']<=20 for v in aligned),
                overhead_min_mm=min(v['distance_mm'] for v in aligned),overhead_end_mm=aligned[-1]['distance_mm'],
                wrist_detections=len(boxed),within_20_and_box=sum(r['grasp_alignment']['distance_mm']<=20 for r in boxed),
                candidate_horizontal_fail=sum(failure(r)['horizontal'] for r in boxed),
                candidate_depth_fail=sum(failure(r)['depth'] for r in boxed),
                xy_net_motion_from_4s_mm=norm(np.asarray(sent[-1]['tip_xyz_mm'])[:2]-np.asarray(last4['tip_xyz_mm'])[:2]),
                plans_total=len(plans),plans_from_4s=len(slow),
                from_4s={n:stats([p[n] for p in slow]) for n in ('raw_mm','before_weight_mm','height_weight','cue_weight','weight','filtered_mm','eight_step_mm','hypothetical_eight_step_mm','along_overhead_mm','hypothetical_along_overhead_mm')},
                dispatch_from_4s={n:stats([p[n] for p in ticks]) for n in ('pred_sent','sent_fb')},
                gain=float(par['gain']),last_2s_chunk_offsets=dict(collections.Counter(p['offsets'] for p in tracking if p['t']>=4.3)),
                snapshots=[{k:round(p[k],5) for k in ('seconds','raw_mm','before_weight_mm','weight','filtered_mm','eight_step_mm')} for p in plans.values()])


if __name__=='__main__':
    print(json.dumps(inspect(Path(sys.argv[1]),sys.argv[2]),ensure_ascii=False))
