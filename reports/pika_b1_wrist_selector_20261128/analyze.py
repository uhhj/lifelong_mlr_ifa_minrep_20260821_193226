"""Read one completed B1 run and compare fixed-input planar targets, without hardware I/O."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def stats(values):
    return {key:round(float(value),6) for key,value in
            zip(('min','median','p95','max'),np.percentile(values,[0,50,95,100]))}


def inspect(work, name, candidate):
    cv2.setNumThreads(1)
    folder=work/'policy_runs'/name
    path=folder.with_suffix('.jsonl')
    rows=read(path)
    source=folder/'inputs.jsonl'
    samples={r['stamp']:r for r in read(source)}
    profile=rows[0]['policy']
    bundle=profile['alignment']
    expected_original='4e995bc1fdc9e56b2219edb30702a5d18e788d05ff7c6ad10925755c029b3a8e'
    expected_candidate='27fca3b2b3252827076d08aff0a122e3ec12004e07959ce7992c0b74e0b3d88d'
    original_id=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert original_id==profile['sha256']==expected_original
    frozen=json.loads((candidate/'profiles.json').read_bytes())[rows[0]['task']]
    config=frozen['alignment']
    assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==frozen['sha256']==expected_candidate
    assert digest(candidate/'ode_alignment.py')==config['backend_sha256']
    for n,h in config['station_files'].items():
        assert digest(candidate/n)==h
    assert config['base_sha256']==bundle['base_sha256'] and config['files']==bundle['files']
    for n,h in bundle['files'].items():
        assert digest(work/n)==h
    camera=json.loads((candidate/'wrist_camera.json').read_bytes())
    camera_file=candidate/'camera_models.json'
    assert digest(camera_file)=='0a04aa609c4cfd79177b232043f90a676d141321891a486965816bff4b82b32d'
    independent=json.loads(camera_file.read_bytes())['independent']['calibration']

    sys.path.insert(0,str(candidate))
    from wrist_cue import ConfirmedCue, modified_step, signed_cue, table_direction
    from b1_alignment_gate import OVERHEAD_TOLERANCE_MM, wrist_block, wrist_centered
    sys.path.insert(0,str((work/bundle['algorithm']).parent))
    import grounded
    grounded.setup(work)
    from fit_gripper import flange
    with np.load(work/bundle['weights']/'private_policy.npz') as f: par=dict(f)
    with np.load(work/bundle['weights']/'private_settle.npz') as f: settle=dict(f)
    unguarded={k:v for k,v in settle.items() if k!='guard'}

    def pose(q):
        tf=flange(np.rad2deg(np.asarray(q,dtype=float)[:6]))
        return tf[:3,3]+tf[:3,:3]@par['tip'], tf[:3,:3]
    dispatched=[r for r in rows if r.get('mode')=='execute' and r.get('sent')]
    first=next(r['mono_ns'] for r in rows if r['event']=='gripper_stable')
    stop=next(r for r in rows if r['event']=='pika_stop_trace')
    obs=ConfirmedCue()
    plans={}
    ticks=[]
    for row in dispatched:
        stamp=row['stamp']; t=(stamp-first)/1e9
        entry=samples[stamp]
        plan=entry['plan']; align=row['grasp_alignment']; head=plan['chunk_stamp']
        assert align['stamp']==stamp and plan['alignment']['applied']
        frame=cv2.imread(str(folder/entry['images']['model']))
        assert frame.shape==(480,1280,3)
        wrist=frame[:,640:].copy()
        box=wrist_block(wrist)
        assert (list(box) if box is not None else None)==align['wrist_box']
        centered=wrist_centered(box,row['q'][6])
        dist=align['distance_mm']
        assert (dist<=OVERHEAD_TOLERANCE_MM and centered)==align['aligned']
        pos,rot=pose(entry['q'])
        assert np.allclose(pos,row['tip_xyz_mm'],atol=.0001)
        z0,ramp,_,_=settle['guard']
        side=None
        if pos[2]<=settle['contact_z']+z0+ramp:
            side='centered' if centered else signed_cue(wrist)
        confirmed=obs.observe(stamp,side)
        if head not in plans:
            assert head==stamp and plan['offset']==0
            q=np.asarray([samples[v]['q'] for v in entry['history']],dtype=np.float32)
            xy,_=pose(q[-1]); xy=xy[:2]
            uv=np.array(align['block_uv'])
            goal=grounded.settle_goal(uv,q,wrist,par,settle,flange)
            prior_goal=grounded.settle_goal(uv,q,wrist,par,unguarded,flange)
            def step(g):
                return grounded.move(par,xy,g,'feedback','heatmap')-xy
            base=step(goal); before=step(prior_goal)
            weight=np.linalg.norm(goal-xy)/max(1e-12,np.linalg.norm(prior_goal-xy))
            assert 0<=weight<1.0001
            alt=modified_step(base,confirmed,rot,camera,float(par['cap']))
            # Independent geometry maps the late, annotated lower image edge onto the table.
            axis=table_direction(rot,independent,'bottom')
            plans[head]=dict(s=t,mode=confirmed,start=xy,step=base,
                             baseline_norm_mm=float(np.linalg.norm(base)),attenuation=float(weight),
                             prior_norm_mm=float(np.linalg.norm(before)),overhead_mm=dist,
                             baseline_signed_mm=float(base@axis),candidate_signed_mm=float(alt@axis),
                             before_attenuation_signed_mm=float(before@axis))
        a=plans[head]
        expected=a['start']+(plan['offset']+1)*a['step']/8
        pred,_=pose(row['p']);sent,_=pose(row['sent']);measured,_=pose(row['measured_at_send'])
        ticks.append(dict(s=t,intent=row['close_intent'],aligned=align['aligned'],boxed=box is not None,
                          centered=centered,within=dist<=OVERHEAD_TOLERANCE_MM,overhead=dist,cue=confirmed,
                          width=row['measured_at_send'][6],goal=row['pika_target_mm'],xy=pos[:2],
                          raw_reconstruction=float(np.linalg.norm(expected-pred[:2])),
                          pred_sent_mm=float(np.linalg.norm(pred[:2]-sent[:2])),
                          sent_feedback_mm=float(np.linalg.norm(sent[:2]-measured[:2])),
                          pair_skew_ms=entry['skew_ns']/1e6,send_age_ms=(row['send_started_ns']-stamp)/1e6))
    heads=[{k:v for k,v in p.items() if k not in ('start','step')} for p in plans.values()]
    late=[r for r in heads if r['s']>=4]
    tail=[r for r in ticks if r['s']>=4]
    t0_wall=rows[0]['wall_ns']+first-rows[0]['mono_ns']
    def stamp_where(key):
        return next((r['s'] for r in ticks if r[key]),None)
    assert max(r['raw_reconstruction'] for r in ticks)<.001
    live=work/'piper_test_console_pika_b1_full_v1'
    registry=json.loads((live/'profiles.json').read_bytes())
    recorded_key=rows[0]['task']
    preview_receipt=f'verified-{recorded_key}.json'
    def entry_status(directory):
        p=json.loads((directory/'profiles.json').read_bytes())[recorded_key]
        evidence=json.loads((directory/preview_receipt).read_bytes())
        return dict(display_version=p['version'],profile=p['sha256'],receipt=evidence['sha256'],
                    evidence_matches=(p['sha256']==evidence['sha256'] and len(evidence['windows'])>=3))
    inventory=dict(active=entry_status(live),offline_copy=entry_status(candidate),
                   candidate_keys_in_active=[k for k,p in registry.items() if p['sha256']==expected_candidate])
    return dict(session=name,execution_start_beijing=datetime.fromtimestamp(t0_wall/1e9,timezone(timedelta(hours=8))).isoformat(),
                log_sha256=digest(path),inputs_sha256=digest(source),actual_profile=profile['sha256'],
                candidate_profile=frozen['sha256'],desktop_inventory=inventory,duration_setting_s=profile['duration'],configured_end=rows[0]['end_mode'],
                stop_after_s=(stop['stop_requested_ns']-first)/1e9,end=rows[-1]['detail'],
                frames=len(ticks),plans=len(heads),phase_counts=Counter(r['gripper_phase'] for r in dispatched),
                intent=sum(r['intent'] for r in ticks),box=sum(r['boxed'] for r in ticks),within=sum(r['within'] for r in ticks),
                wrist_centered=sum(r['centered'] for r in ticks),passed=sum(r['aligned'] for r in ticks),
                intent_and_box=sum(r['intent'] and r['boxed'] for r in ticks),first_intent_s=stamp_where('intent'),
                first_within_s=stamp_where('within'),last_box_s=max(r['s'] for r in ticks if r['boxed']),
                overhead_start_mm=ticks[0]['overhead'],overhead_end_mm=ticks[-1]['overhead'],overhead_min_mm=min(r['overhead'] for r in ticks),
                goals_mm=[r['goal'] for r in ticks if r['goal'] is not None],width_mm=stats([r['width'] for r in ticks]),
                close_cues_on_fixed_frames=Counter(r['cue'] or 'baseline' for r in ticks),
                first_direction_confirmation_s=stamp_where('cue'),
                max_reconstruction_mm=max(r['raw_reconstruction'] for r in ticks),
                full_send_age_ms=stats([r['send_age_ms'] for r in ticks]),pair_skew_ms=stats([r['pair_skew_ms'] for r in ticks]),
                late_frames=len(tail),late_plans=len(late),late_mode=Counter(r['mode'] or 'baseline' for r in late),
                late_measured_net_mm=float(np.linalg.norm(tail[-1]['xy']-tail[0]['xy'])),
                late_commands={k:stats([r[k] for r in tail]) for k in ('pred_sent_mm','sent_feedback_mm')},
                late_plan_stats={k:stats([r[k] for r in late]) for k in ('overhead_mm','baseline_norm_mm','attenuation',
                                'prior_norm_mm','before_attenuation_signed_mm','baseline_signed_mm','candidate_signed_mm')},
                plan_evidence=[{k:(round(v,6) if isinstance(v,float) else v) for k,v in row.items()} for row in heads])


if __name__=='__main__':
    print(json.dumps(inspect(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])),ensure_ascii=False))
