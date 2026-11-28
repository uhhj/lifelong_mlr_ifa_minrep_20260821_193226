"""Field replay and software STOP verification on archived observations and virtual devices."""
from collections import Counter
from unittest.mock import patch
from pathlib import Path
import sys
import cv2
import numpy as np
from replay import (here,work,old,read,lines,dump,models,reset,blind_records,valid,at,endpoint,eligible,sha,load_record,imaging)


def field(two):
    from runner import GraspIntent
    from b1_placement import B1Placement
    result=[]
    files=sorted((work/'policy_runs').glob('20261128_14*_B1-ode50-wrist-20261127.jsonl'))
    selected=[f for f in files if f.name.startswith(('20261128_142711_','20261128_143131_','20261128_143325_'))]
    assert len(selected)==3
    for file in selected:
        events=lines(file)
        tx=[r for r in events if r.get('mode')=='execute' and r.get('sent')]
        folder=file.with_suffix('')
        inputs={r['stamp']:r for r in lines(folder/'inputs.jsonl')}
        first=tx[0];warm=inputs[first['stamp']]['history'][:-1]
        versions=[]
        for idx,m in enumerate(two):
            reset(m);gate=B1Placement();intent=GraspIntent()
            flags=[];width=[];tracking=[];observer=[]
            prev=first['measured_at_send'];last=0
            for entry,row in [(inputs[t],None) for t in warm]+[(inputs[r['stamp']],r) for r in tx]:
                stamp=entry['stamp'];q=entry['q'];im=cv2.imread(str(folder/entry['images']['model']))
                active=row is not None
                if active and row is not first:
                    prev=prior['sent'];last=prior['stamp']
                query=dict(q=q,execution=first['gate_execution'] if active else 0,mode_at_query='execute' if active else 'preview',
                           previous_sent=prev if active else None,previous_stamp=last,gripper_phase='approach')
                m.predict_recorded(m.prepare_views([im[:,:640],im[:,640:]]),query,stamp)
                if not active:continue
                prior=row
                assert m.trace['ready']
                flags.append(m.visual['aligned'])
                if idx==0:assert flags[-1]==row['grasp_alignment']['aligned']
                if idx==1:
                    observer.append(m.relative.target is not None)
                    if m.relative.target is not None:tracking.append(np.linalg.norm(m.relative.target-m.aligner.fk(q)))
                intent.observe(stamp,q[6],stamp,m.intent_widths)
                goal=gate.next_goal(stamp,intent.fired,row['tip_xyz_mm'],row['measured_at_send'][6],vision=m.visual)
                if goal is not None:width.append(goal)
            versions.append(dict(aligned_indices=[i+1 for i,x in enumerate(flags) if x],simulated_width_goals=width,
                                 tracked_wrist_ticks=sum(observer),median_wrist_relative_mm=round(float(np.median(tracking)),2) if tracking else None))
        result.append(dict(session=file.stem,frames=len(tx),versions=versions,raw_sha256=sha(file)))
    return result


def height_windows(two):
    data=blind_records();labels=read(here/'blind_labels.json')
    gt=np.array(read(work/'offline_diagnostics/b1_vision_contrasts_20261124/final.json')['geometry']['table_map'])
    m=two[0];out={}
    height=float(m.aligner.steady['contact_z'])+30.
    for e,r in data.items():
        goal=np.r_[labels[e],1]@gt;found=None
        for i,row in enumerate(r['rows']):
            if not valid(row) or i<7 or (r['close'] is not None and i>=r['close']):continue
            q=at(row,'feedback');t=m.aligner.flange(np.rad2deg(q[:6]));z=(t[:3,3]+t[:3,:3]@m.aligner.par['tip'])[2]
            aux=dict(r,close=i+18)
            if z<=height and eligible(aux) and q[6]>55:
                found=i;break
        steps=[]
        if found is not None:
            for m in two:
                end,start=endpoint(m,aux,aux)
                steps.append(round(float(np.linalg.norm(start-goal)-np.linalg.norm(end-goal)),4))
        out[e]=dict(frame=found,approach_mm=steps)
    return dict(threshold_z_mm=height,episodes=out)


def lost_closes(model):
    from relative_xy import wrist_measurement
    selection=read(work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json')
    ranking={e['id']:e for e in selection['ranking']}
    counts=Counter()
    for ep,r in read(here/'regression.json')['episodes'].items():
        old_event,new_event=r['close_events']
        if not old_event or new_event:continue
        entry=ranking[ep]
        rec=load_record(work/'batches/B1_20261123'/entry['source'],entry['first'])
        i=old_event['frame'];reset(model)
        # Last 35 frames exceed the 8-image history and 300 ms stationary-target memory.
        for j in range(max(0,i-34),i+1):
            row=rec['rows'][j];q=at(row,'feedback');stamp=row['sample_monotonic_ns']
            prev=at(rec['rows'][max(0,j-1)],'target')
            views=[imaging(rec,j,role) for role in ('overhead','wrist')]
            model.predict_recorded(model.prepare_views(views),dict(q=q,execution=1,mode_at_query='execute',previous_sent=prev,
                                   previous_stamp=model.last_stamp,gripper_phase='approach'),stamp)
        uv=model.visual['block_uv']
        coarse=np.r_[uv,1.]@model.aligner.par['matrix']
        rot=model.pose[:3,:3];tip=model.pose[:3,3]+rot@model.aligner.par['tip']
        face=wrist_measurement(views[1],tip,rot,coarse,model.relative.calibration)
        if face is None:key='no_current_complete_face'
        elif model.relative.target is None:key='unstable_or_insufficient_recent_metric_centers'
        elif not model.relative.centered:key='estimated_center_outside_clearance'
        elif not model.visual['aligned']:key='old_gate_not_aligned_at_replay'
        else:key='other_timing_or_three_frame_confirmation'
        counts[key]+=1
    return dict(counts)


if __name__=='__main__':
    cv2.setNumThreads(1)
    if len(sys.argv)>3 and sys.argv[3]=='coverage':
        from coverage_check import measure
        measure()
        sys.exit(0)
    two=models()
    import guard_case,runner
    key='B1-ode50-wrist-20261127'
    p=read(here/'candidate.json')
    cases=['preview_only','return_pending','unknown_mode','confirm_only','outside_pose','usb_unready',
           'success','failed','stop','close','deadline','release','heartbeat','stale_prediction','no_stop_ack']
    # Field station is deliberately unmodified. It must reject the offline-only identity.
    try:runner.uses_fixed_grip(key,p)
    except ValueError:pass
    else:raise AssertionError('Field registry unexpectedly accepts offline identity')
    saved=runner.PROFILES[key]
    assert p['duration']==saved['duration'] and p['end_mode']==saved['end_mode']=='stop'
    checks=[guard_case.exercise(c,key) for c in cases]
    dump('checks.json',dict(cases=checks,offline_identity_rejected_by_live_registry=True,deadline_s=p['duration'],heartbeat_ns=runner.SAFETY_NS,freshness_ns=runner.FRESH_NS,
                            field=field(two),height_windows=height_windows(two),lost_regression_close_breakdown=lost_closes(two[1])))
    print('verified guards and fixed-feedback field replay')
