"""Compare candidate and archived B1 plans on identical observations. All I/O is from disk."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

from b1_placement import B1Placement
from b1_alignment_gate import wrist_block, wrist_centered
from bus import TargetGuardFault, bounded_target
from catalog import catalog
from runner import B1_FIXED_GRIP, GraspIntent, load_policy
from calibrate import references
from wrist_cue import ConfirmedCue, modified_step, signed_cue, table_direction

HERE = Path(__file__).resolve().parent
RUNS = [('223635_725397', 'right', 3.68), ('224525_730400', 'bottom', 4.0)]
SIDE = ('left','right','bottom')


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if not values:
        return None
    return {k:round(float(v),5) for k,v in zip(('min','median','max'), np.percentile(values,[0,50,100]))}


def reset(model):
    for queue in (model.history, model.raw, model.heat, model.features):
        queue.clear()
    model.execution = None
    model.last_stamp = 0
    model.chunk = model.last_result = None
    model.cue = ConfirmedCue()
    model.mode = model.plan_mode = None


def positions(model, q):
    return model.aligner.xy.pose(np.asarray(q)[...,:6],model.aligner.par['tip'])[0]


def field(work, model, key, label, lo, independent):
    path=next((work/'policy_runs').glob('20261127_'+key+'_B1-*.jsonl'))
    folder=path.with_suffix('')
    log=rows(path)
    sent=[r for r in log if r.get('sent') and r.get('mode')=='execute']
    source={r['stamp']:r for r in rows(folder/'inputs.jsonl')}
    start=next(r['mono_ns'] for r in log if r['event']=='gripper_stable')
    align=model.aligner
    def get_image(r):
        pair=cv2.imread(str(folder/r['images']['model']))
        assert pair.shape==(480,1280,3)
        return pair[:,:640],pair[:,640:]
    state=ConfirmedCue()
    augmented=[]
    plan_info={}
    fixed=[]
    triggers=Counter()
    for row in sent:
        stamp=row['stamp']; entry=source[stamp]; plan=entry['plan']
        a=plan['alignment']
        pose=align.flange(np.rad2deg(entry['q'][:6]))
        tip=pose[:3,3]+pose[:3,:3]@align.par['tip']
        z0,ramp,_,_=align.steady['guard']
        raw=None
        if a['applied'] and tip[2]<=align.steady['contact_z']+z0+ramp:
            wrist=get_image(entry)[1]
            box=wrist_block(wrist)
            if wrist_centered(box,entry['q'][6]):
                raw='centered'
            else:
                raw=signed_cue(wrist)
        mode=state.observe(stamp,raw)
        triggers[mode or 'baseline']+=1
        head=plan['chunk_stamp']
        if head not in plan_info:
            h=source[head]; ha=h['plan']['alignment']
            if not ha['applied']:continue
            q=np.asarray([source[t]['q'] for t in h['history']],dtype=np.float32)
            xy=align.fk(q[-1]); uv=ha['block_uv']
            goal=model.algorithm.settle_goal(uv,q,get_image(h)[1],align.par,align.steady,align.flange)
            step=model.algorithm.move(align.par,xy,goal,'feedback','heatmap')-xy
            rot=align.flange(np.rad2deg(q[-1,:6]))[:3,:3]
            t=(head-start)/1e9
            valid=mode if head==stamp else None
            changed=modified_step(step,valid,rot,model.camera,float(align.par['cap']))
            assert np.linalg.norm(changed-step)<2.00001 or np.linalg.norm(changed)<=np.linalg.norm(step)+1e-8
            if t>=lo and head==stamp:
                ref=table_direction(rot,independent,label)
                axis=table_direction(rot,model.camera,label)
                assert ref is not None and axis is not None
                fixed.append(dict(s=round(t,4),mode=valid,old_mm=round(float(step@ref),5),
                                  new_mm=round(float(changed@ref),5),axis_cos=round(float(ref@axis),6)))
            plan_info[head]=(xy,step)
        xy,step=plan_info[head]
        archived=positions(model,row['p'])[:2]
        augmented.append(float(np.linalg.norm(archived-(xy+(plan['offset']+1)*step/8))))
    assert max(augmented)<.01 and fixed
    def replay(guarded):
        reset(model)
        intent,placement=GraspIntent(),B1Placement()
        tick=sent[0]['gate_execution'];prev=sent[0]['sent'][:]
        prev[:6]=sent[0]['measured_at_send'][:6]
        last_ack=0;last_send=tick;seen=set()
        widths=[];offsets=Counter();plans=Counter();times=[];zerr=[];widtherr=[]
        interruption=None
        seed=[source[t] for t in source[sent[0]['stamp']]['history'][:-1]]
        for entry,row in [(e,None) for e in seed]+[(source[r['stamp']],r) for r in sent]:
            active=row is not None;t=entry['stamp'];q=entry['q']
            pix=model.prepare_views(get_image(entry))
            query=dict(q=q,execution=tick if active else 0,mode_at_query='execute' if active else 'preview',
                       previous_stamp=last_ack,previous_sent=prev if active else None,gripper_phase=placement.phase)
            then=time.perf_counter()
            target=model.predict_recorded(pix,query,t)
            elapsed=(time.perf_counter()-then)*1000
            if not active:continue
            times.append(elapsed);p=model.trace
            assert p['ready']
            offsets[p['offset']]+=1
            if p['chunk_stamp'] not in seen:
                seen.add(p['chunk_stamp']);plans[model.plan_mode or 'baseline']+=1
                dz=positions(model,model.chunk)[:,2]-positions(model,model.base_plan)[:,2]
                zerr.extend(abs(dz).tolist())
                widtherr.extend(abs(model.chunk[:,6]-model.base_plan[:,6]).tolist())
            intent.observe(t,q[6],t,model.intent_widths)
            measured=row['measured_at_send']
            now=row['send_started_ns']
            if guarded:
                try:
                    arm=bounded_target(target,measured,prev,(now-last_send)/1e9,b3_tracking=True,arm_only=True)
                except TargetGuardFault as exc:
                    interruption=dict(s=round((t-start)/1e9,4),axes=[v['axis'] for v in exc.violations])
                    break
            else:
                arm=target[:6]
            next_goal=placement.next_goal(t,intent.fired,row['tip_xyz_mm'],measured[6],vision=model.visual)
            if next_goal is not None:
                assert next_goal!=35 or model.visual['aligned'] and placement.aligned_ticks>=3
                widths.append(next_goal)
            prev=[*arm,prev[6] if next_goal is None else next_goal]
            last_send=now;last_ack=t
        assert max(zerr)<.001 and not max(widtherr)
        return dict(frames=len(times),plan_modes=plans,offsets=offsets,grip_targets_mm=widths,guard_stop=interruption,
                    max_z_change_mm=max(zerr),max_width_change_mm=max(widtherr),p95_ms=round(float(np.percentile(times,95)),2))
    return dict(session=path.stem,log_sha256=sha(path),inputs_sha256=sha(folder/'inputs.jsonl'),
                recorded_frames=len(sent),source_end=log[-1].get('outcome'),end_detail=log[-1].get('detail'),
                reference_axis=label,reference_start_s=lo,
                confirmed_frames=triggers,verified_original_targets=len(augmented),original_reconstruction_max_mm=max(augmented),
                fixed_heads=fixed,baseline_signed_mm=stats([v['old_mm'] for v in fixed]),
                candidate_signed_mm=stats([v['new_mm'] for v in fixed]),
                plan_delta_signed_mm=stats([v['new_mm']-v['old_mm'] for v in fixed]),
                independent_axis_cos=stats([v['axis_cos'] for v in fixed]),schedule=replay(False),guarded=replay(True))


def controls(model,work):
    counts={'fit':Counter(),'other':Counter()}
    samples={'fit':[],'other':[]}
    for lab,record in references(work):
        category='fit' if lab['train'] else 'other'
        folder=work/'batches/B1_20261123'/record['source']
        data=rows(folder/'alignment.jsonl')[record['first']:]
        c=lab['close']
        if c>=len(data):continue
        lo=max(0,c-24);hi=min(len(data),c+7)
        reset(model)
        detector,gate=GraspIntent(),B1Placement()
        seen=False; moved=False; close=False;quiet=False;wrong=False
        close_tick=None
        raw_at_expert_close=None
        early_dirs=[]
        contact=lab.get('contact')
        ref_q=data[c if contact is None else contact]['feedback']['joints_rad']
        destination=positions(model,ref_q)[:2]
        for j in range(lo,hi):
            r=data[j];stamp=r['sample_monotonic_ns']
            q=r['feedback']['joints_rad']+[r['feedback']['gripper_mm']]
            prev=data[max(0,j-1)]['target']['joints_rad']+[data[lo]['feedback']['gripper_mm']]
            active=j>=lo+7
            query=dict(q=q,execution=int(active),mode_at_query='execute' if active else 'preview',
                       previous_sent=prev if active else None,previous_stamp=model.last_stamp,gripper_phase=gate.phase)
            views=[cv2.imread(str(folder/'episode0/camera/color'/name/f'{stamp/1e9:.6f}.jpg')) for name in ('overhead','wrist')]
            model.predict_recorded(model.prepare_views(views),query,stamp)
            if not active:continue
            is_close_frame=j==c
            m=model.mode
            if is_close_frame:
                wrong=m in SIDE
                raw_at_expert_close=signed_cue(views[1])
            if m in SIDE and j<c and model.trace['alignment'].get('applied'):
                moved=True
                if model.trace['offset']==0:
                    d=destination-positions(model,q)[:2]
                    if np.linalg.norm(d)>2:
                        dr=table_direction(model.pose[:3,:3],model.camera,m)
                        if dr is not None:early_dirs.append(float(d@dr))
            if m=='centered':
                quiet=True
                rec=model.trace['alignment']
                if 'revised_xy_mm' in rec:
                    assert np.linalg.norm(rec['revised_xy_mm'])<=.50001
            detector.observe(stamp,q[6],stamp,model.intent_widths)
            seen |= model.visual['aligned']
            goal=gate.next_goal(stamp,detector.fired,positions(model,q),q[6],vision=model.visual)
            if goal==35:
                assert model.visual['aligned'] and gate.aligned_ticks>=3
                close=True
                close_tick=j-c
        outcomes=dict(centered=seen,center_cap=quiet,augmented_before_expert_close=moved,
                      augmented_at_expert_close=wrong,close_accepted=close)
        counts[category].update(windows=1,**{k:int(v) for k,v in outcomes.items()})
        samples[category].append(dict(episode=lab['episode'],**outcomes,close_tick=close_tick,
                                      raw_side_at_expert_close=raw_at_expert_close,expert_projection_mm=stats(early_dirs)))
    return dict(counts=counts,episodes=samples)


def main(work):
    cv2.setNumThreads(1);torch.set_num_threads(2)
    profile=catalog()[B1_FIXED_GRIP]
    model=load_policy(profile)
    geometry=json.loads((HERE/'camera_models.json').read_bytes())
    result=dict(profile=B1_FIXED_GRIP,sha256=profile['sha256'],duration=profile['duration'],
                end_mode=profile['end_mode'],camera_file_sha256=sha(HERE/'wrist_camera.json'),
                calibration={k:{a:v[a] for a in ('own','opposite')} for k,v in geometry.items()})
    result['field']=[]
    for key,label,lo in RUNS:
        item=field(work,model,key,label,lo,geometry['independent']['calibration'])
        result['field'].append(item)
        print(key,item['baseline_signed_mm'],item['candidate_signed_mm'],item['guarded'],flush=True)
    (HERE/'field_result.json').write_text(json.dumps(result,indent=2)+'\n')
    result['controls']=controls(model,work)
    print('controls',result['controls']['counts'],flush=True)
    live=work/'piper_test_console_pika_b1_full_v1'
    original=json.loads((live/'profiles.json').read_bytes())[B1_FIXED_GRIP]
    assert sha(live/'ode_alignment.py')==original['alignment']['backend_sha256']
    spec=importlib.util.spec_from_file_location('reference_ode',live/'ode_alignment.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline=controls(module.GroundedODE(original),work)
    result['baseline_expert_counts']=baseline['counts']
    result['active_station']=dict(sha256=original['sha256'],end_mode=original['end_mode'],duration=original['duration'])
    result['shared_station_files']={}
    for n in ('runner.py','ode_live.py','b1_alignment_gate.py','b1_placement.py','bus.py','b1_gripper.py','catalog.py','desktop.py'):
        assert sha(HERE/n)==sha(live/n)
        result['shared_station_files'][n]=sha(live/n)
    result['expert_close_differences']=[]
    for group in baseline['episodes']:
        for a,b in zip(baseline['episodes'][group],result['controls']['episodes'][group]):
            assert a['episode']==b['episode']
            if a['close_tick']!=b['close_tick']:
                result['expert_close_differences'].append(a['episode'])
    (HERE/'outcomes.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
