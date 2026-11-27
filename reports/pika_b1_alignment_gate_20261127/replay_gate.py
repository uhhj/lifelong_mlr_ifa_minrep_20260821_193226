"""Replay archived camera pairs and feedback without constructing any device or GUI."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from b1_placement import B1Placement
from bus import TargetGuardFault, bounded_target
from catalog import HERE, catalog
from grip_window import BUNDLE
from runner import B1_FIXED_GRIP, B1_RETURN_PENDING, GraspIntent, SAFETY_NS, Session, load_policy


def rows(path):
    with path.open(encoding='utf8') as source:
        return [json.loads(line) for line in source]


def reset(model):
    model.history.clear()
    model.raw.clear()
    model.heat.clear()
    model.features.clear()
    model.execution = None
    model.last_stamp = 0
    model.chunk = model.last_result = None


def guard_checks(model, pixels, query, stamp):
    import math
    from types import SimpleNamespace
    try:
        model.predict_recorded(pixels, dict(query,previous_stamp=stamp-1), stamp+33_333_333)
    except ValueError as exc:
        assert '确认' in str(exc)
    else:
        raise AssertionError('missing acknowledgement accepted')
    q = [0., .8, -.8, 0., 1., 0., 60.]
    outside = q.copy()
    outside[1] += math.radians(6)
    try:
        bounded_target(q,outside,q,.033,b3_tracking=True,arm_only=True)
    except TargetGuardFault:
        pass
    else:
        raise AssertionError('joint tracking guard bypassed')
    session = Session(SimpleNamespace(active=True),None,None,0)
    try:
        session.watchdog(SAFETY_NS)
    except RuntimeError as exc:
        assert '500' in str(exc)
    else:
        raise AssertionError('heartbeat did not stop')
    session.aligned_b1=True
    try:
        session.begin(0)
    except ValueError as exc:
        assert str(exc)==B1_RETURN_PENDING
    else:
        raise AssertionError('onsite start lock lost')


def replay(model, folder, guarded):
    old = [r for r in rows(folder.with_suffix('.jsonl')) if 'sent' in r and r.get('mode')=='execute']
    inputs = {r['stamp']:r for r in rows(folder/'inputs.jsonl')}
    tick = old[0]['gate_execution']
    seed = [inputs[s] for s in inputs[old[0]['stamp']]['history'][:-1]]
    reset(model)
    detect, gate = GraspIntent(), B1Placement()
    previous = old[0]['sent'][:]
    previous[:6]=old[0]['measured_at_send'][:6]
    prev_stamp=0
    last_send=tick
    offsets=collections.Counter()
    max_height_error=0.
    corrected=0
    intent_ticks=0
    at_old_close=[]
    writes=[]
    stop=None
    times=[]
    for rec, original in [(r,None) for r in seed]+[(inputs[r['stamp']],r) for r in old]:
        stamp=rec['stamp'];active=original is not None
        q=rec['q']
        im=cv2.imread(str(folder/rec['images']['model']))
        assert im.shape==(480,1280,3)
        pix=model.prepare_views((im[:,:640],im[:,640:]))
        query=dict(q=q,mode_at_query='execute' if active else 'preview',execution=tick if active else 0,
                   previous_sent=previous if active else None,previous_stamp=prev_stamp,gripper_phase=gate.phase)
        start=time.perf_counter()
        pred=model.predict_recorded(pix,query,stamp)
        times.append(1000*(time.perf_counter()-start))
        if not active:continue
        plan=model.trace
        assert plan['ready']
        i=plan['offset']; offsets[i]+=1
        evidence=plan['alignment']
        assert evidence['stamp']==stamp
        if evidence['applied']:
            corrected+=1
            for joint_seq in (model.chunk, model.base_plan):
                assert joint_seq.shape==(8,7)
            T=model.aligner.flange(np.rad2deg(np.vstack((model.chunk[:,:6],model.base_plan[:,:6]))))
            zz=(T[:,:3,3]+T[:,:3,:3]@model.aligner.par['tip'])[:,2]
            max_height_error=max(max_height_error,float(np.max(abs(zz[:8]-zz[8:]))))
        detect.observe(stamp,q[6],stamp,model.intent_widths)
        xyz=original['tip_xyz_mm']
        if detect.fired:intent_ticks+=1
        if original['pika_target_mm']==35:
            at_old_close.append({'aligned':evidence['aligned'],'distance_mm':evidence['distance_mm'],
                                 'wrist_box':evidence['wrist_box'],'phase':gate.phase})
        measured=original['measured_at_send']
        if guarded:
            now=original['send_started_ns']
            try:
                arm=bounded_target(pred,measured,previous,(now-last_send)/1e9,b3_tracking=True,arm_only=True)
            except TargetGuardFault as exc:
                stop={'seconds':round((stamp-tick)/1e9,3),'axes':[v['axis'] for v in exc.violations]}
                break
            last_send=now
        else:
            # Isolate chunk reuse: six synthetic acknowledgements mirror the
            # candidate output. Inputs and measured jaw width remain recorded.
            arm=pred[:6]
        goal=gate.next_goal(stamp,detect.fired,xyz,measured[6],vision=evidence)
        if goal is not None:
            if goal==35:assert evidence['aligned'] and gate.aligned_ticks>=3
            writes.append([round((stamp-tick)/1e9,3),goal])
        previous=[*arm,previous[6] if goal is None else goal]
        prev_stamp=stamp
    if not guarded:
        guard_checks(model,pix,query,stamp)
    assert max_height_error<.001
    return dict(offset_counts=dict(sorted(offsets.items())),corrected_frames=corrected,intent_ticks=intent_ticks,
                old_close=at_old_close,gripper_events=writes,guard_stop=stop,
                max_plan_z_change_mm=max_height_error,model_p95_ms=round(float(np.percentile(times[7:],95)),2))


def positive_controls(model, work):
    ref=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    labels=json.loads((ref/'points.json').read_bytes())
    for k,v in json.loads((ref/'annotations.json').read_bytes()).items():
        labels[int(k)].update(v)
    manifest=json.loads((work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json').read_bytes())
    ranking={r['id']:r for r in manifest['ranking']}
    results=collections.Counter()
    for lab in labels:
        if not lab['train'] or lab['close'] is None:continue
        e=ranking[lab['episode']];folder=work/'batches/B1_20261123'/e['source']
        source=rows(folder/'alignment.jsonl')[e['first']:]
        c=lab['close']
        lo=max(0,c-24); hi=min(c+7,len(source))
        reset(model);detector=GraspIntent();gate=B1Placement()
        has_intent=False; visual=False; sent=False;streak=0
        for j in range(lo,hi):
            r=source[j];stamp=r['sample_monotonic_ns']
            q=r['feedback']['joints_rad']+[r['feedback']['gripper_mm']]
            target=source[max(0,j-1)]['target']
            previous=target['joints_rad']+[source[lo]['feedback']['gripper_mm']]
            active=j>=lo+7
            query=dict(q=q,execution=1 if active else 0,mode_at_query='execute' if active else 'preview',
                       previous_sent=previous if active else None,previous_stamp=model.last_stamp,
                       gripper_phase=gate.phase)
            views=[cv2.imread(str(folder/'episode0/camera/color'/role/f'{stamp/1e9:.6f}.jpg')) for role in ('overhead','wrist')]
            model.predict_recorded(model.prepare_views(views),query,stamp)
            if not active:continue
            detector.observe(stamp,q[6],stamp,model.intent_widths)
            has_intent |= detector.fired
            pose=model.aligner.flange(np.rad2deg(q[:6]))
            xyz=pose[:3,3]+pose[:3,:3]@model.aligner.par['tip']
            goal=gate.next_goal(stamp,detector.fired,xyz,q[6],vision=model.visual)
            visual |= gate.aligned_ticks>=3
            if goal==35:
                assert model.visual['aligned'] and detector.fired
                sent=True
                break
        results.update(windows=1,with_close_intent=int(has_intent),with_visual_confirmation=int(visual),accepted=int(sent))
    assert results['accepted']>0
    return dict(results)


def main(work):
    key=B1_FIXED_GRIP
    profile=catalog()[key]
    assert profile['sha256']==BUNDLE
    cv2.setNumThreads(1); torch.set_num_threads(2)
    model=load_policy(profile)
    gate=B1Placement()
    for i,b in enumerate([True,True,False,True,True]):
        t=(i+1)*33_333_333
        assert gate.next_goal(t,True,[0,0,0],60,vision={'stamp':t,'aligned':b}) is None
    for t,obs in [(300_000_000,300_000_000),(333_333_333,300_000_000),(366_666_666,366_666_666),
                  (400_000_000,400_000_000)]:
        assert gate.next_goal(t,True,[0,0,0],60,vision={'stamp':obs,'aligned':True}) is None
    assert gate.next_goal(433_333_333,True,[0,0,0],60,vision={'stamp':433_333_333,'aligned':True})==35
    assert gate.next_goal(466_666_666,True,[0,0,0],36,vision={}) is None
    assert gate.phase=='carrying'
    windows=[]
    for session in ('212732_689056','213025_690417','213755_694001'):
        path=next((work/'policy_runs').glob('20261127_'+session+'_B1-*.jsonl'))
        record={'session':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for label in ('schedule','with_tracking_guard'):
            record[label]=replay(model,path.with_suffix(''),label=='with_tracking_guard')
            print(session,label,record[label],flush=True)
        assert record['schedule']['offset_counts'].get(7,0)>0 and not record['schedule']['gripper_events']
        assert not record['with_tracking_guard']['gripper_events']
        assert record['schedule']['corrected_frames']>0
        windows.append(record)
    controls=positive_controls(model,work)
    print('expert controls',controls,flush=True)
    out=dict(profile=key,sha256=BUNDLE,windows=windows,expert_controls=controls,
             checks=['fresh_timestamp','lost_alignment','frame_gap','one_shot_close','arm_acknowledgement',
                     'joint_tracking_stop','heartbeat_500_ms','existing_B1_start_lock'])
    (HERE/'replay_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]))
