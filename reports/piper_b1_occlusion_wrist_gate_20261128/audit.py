"""Continuous-history eight-step replay, occlusion tracking errors, and virtual STOP controls."""
from basis import *
def full_endpoint(m,rec):
    rp.reset(m);j=rec['close']-18;bad=0;held=0;unobserved=0;wrist=0;reference_errors=[]
    truth=rec.get('truth_uv')
    for i,row in enumerate(rec['rows'][:j+1]):
        if not rp.valid(row):
            rp.reset(m);continue
        q=rp.at(row,'feedback')
        past=rp.at(rec['rows'][i-1],'target') if i and rp.valid(rec['rows'][i-1]) else q
        if i==j:m.chunk=None
        stamp=row['sample_monotonic_ns']
        imgs=[rp.imaging(rec,i,k) for k in ('overhead','wrist')]
        m.predict_recorded(m.prepare_views(imgs),dict(q=q,execution=1,mode_at_query='execute',previous_sent=past,
                            previous_stamp=m.last_stamp,gripper_phase='approach'),stamp)
        if truth is not None and m.visual.get('detected_uv') is not None and 'track_mode' in m.visual:
            if np.linalg.norm(np.asarray(m.visual['detected_uv'])-truth)>45:
                bad+=1
                if m.visual.get('wrist_guide'):
                    wrist+=1
                    reference_errors.append(float(np.linalg.norm(m.relative.wrist_goal-rec['goal'])))
                elif not m.visual.get('guide'):unobserved+=1
                elif m.visual['tracker_uv'] is not None and np.linalg.norm(np.asarray(m.visual['tracker_uv'])-truth)<=20:held+=1
    assert m.trace['ready'] and m.trace['offset']==0
    a=m.aligner
    def pose(plan):return a.xy.pose(np.asarray(plan)[:,:6].astype(float),a.par['tip'])
    xyz,rot=pose(m.chunk);base,r0=pose(m.base_plan)
    assert np.max(abs(xyz[:,2]-base[:,2]))<.001 and np.max(abs(rot-r0))<1e-5 and np.array_equal(m.chunk[:,6],m.base_plan[:,6])
    trace=m.trace['alignment']
    return xyz[-1,:2],a.fk(q),dict(track=trace.get('track_mode'),phase=trace['phase'],raw=trace.get('detected_uv'),
           used=trace.get('tracker_uv'),wrist=trace.get('wrist_guide'),jump_ticks=bad,withheld_ticks=unobserved,
           wrist_ticks=wrist,wrist_max_error=None if not reference_errors else round(max(reference_errors),3),correct_ticks=held)
def natural(group):
    baseline=setup();candidate=model(dest,'ode_tracked.py')
    if group=='archive':
        truth={r['ep']:r for r in read(edge/'diag.json')['test']}
        recs=dict(rows('test'));goals={e:np.array(x['goal']) for e,x in truth.items()}
        for e,r in recs.items():r['truth_uv']=truth[e]['label'];r['goal']=goals[e]
    else:
        from assess import reviewed
        prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
        gt=np.array(read(prior/'final.json')['geometry']['table_map'])
        ids=set(read(edge/'regression.json')['episodes'])
        labels=[r for r in reviewed(work,prior) if r['episode'] in ids]
        recs={r['episode']:dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in labels}
        goals={r['episode']:np.r_[r['uv'],1.]@gt for r in labels}
    result={}
    for idx,(ep,r) in enumerate(recs.items()):
        out=[full_endpoint(m,r) for m in (baseline,candidate)]
        result[ep]=dict(natural=[round(float(np.linalg.norm(start-goals[ep])-np.linalg.norm(end-goals[ep])),4) for end,start,_ in out],
                        tracking=out[1][2])
        if group=='archive':
            for a in ('raw','used'):
                uv=out[1][2][a];result[ep][a+'_error_px']=None if uv is None else round(float(np.linalg.norm(np.asarray(uv)-r['truth_uv'])),2)
        if (idx+1)%10==0:print('continuous',group,idx+1,flush=True)
    dump('continuous_'+group+'.json',result)
def controls():
    setup()
    live_old=rp.models()[0];m=model(dest,'ode_tracked.py')
    import verify,guard_case,runner
    p=read(dest/'candidate.json');key='B1-ode50-wrist-20261127'
    assert p['end_mode']==runner.PROFILES[key]['end_mode']=='stop'
    try:runner.uses_fixed_grip(key,p)
    except ValueError:pass
    else:raise AssertionError('Offline identity unexpectedly accepted')
    names=['preview_only','return_pending','unknown_mode','confirm_only','outside_pose','usb_unready','success','failed','stop','close','deadline','release','heartbeat','stale_prediction','no_stop_ack']
    output=dict(cases=[guard_case.exercise(n,key) for n in names],heartbeat_ns=runner.SAFETY_NS,freshness_ns=runner.FRESH_NS,
                field=verify.field([live_old,m]))
    dump('controls.json',output)
    print('controls',len(output['cases']),[r['versions'][1]['simulated_width_goals'] for r in output['field']],flush=True)
def remaining():
    from unittest.mock import patch
    setup();m=model(dest,'ode_tracked.py')
    from b1_placement import B1Placement
    from b1_alignment_gate import wrist_centered
    prior=read(edge/'misses.json');outcomes=read(dest/'archive.json')['episodes']
    ids={e for e,r in prior.items() if r['group']=='archive' and r['reason']=='no_wrist_confirmation' and not outcomes[e]['close_events'][1]}
    gate=B1Placement.next_goal;output={}
    for e,rec in rows('test'):
        if e not in ids:continue
        stamps={r['sample_monotonic_ns']:i-rec['close'] for i,r in enumerate(rec['rows'])}
        frames=[]
        def observe(obj,stamp,intent,xyz,width,*,vision):
            i=stamps[stamp]
            if -8<=i<=8:
                box=m.relative.line_box
                xok=yok=size=None
                if box is not None and vision['view'] is not None:
                    x,y,w,h=box
                    xok=bool(abs(x+w/2-355)<=min(42.,max(0.,(width-35.)/2-2.)*w/35.) and width>39)
                    yok=bool(276<=y<=372 and y+h>=405)
                    size=bool(w>=m.widths[vision['view']]['min_width'])
                frames.append(dict(i=i,intent=bool(intent),coarse=bool(vision['close_coarse'] and vision['distance_mm']<=20),height=bool(vision['depth_ok']),
                    lines=box is not None,stable=len(m.relative.corners)>=3,jaw_x=xok,jaw_depth=yok,size=size,
                    legacy_metric=bool(vision['fresh_metric_center'] and wrist_centered(vision['wrist_box'],width)),
                    visual_pass=bool(vision['aligned'])))
            return gate(obj,stamp,intent,xyz,width,vision=vision)
        with patch.object(B1Placement,'next_goal',observe):
            assert rp.close_event(m,rec) is None
        assert [x['i'] for x in frames]==list(range(-8,9))
        output[e]={k:''.join('?' if f[k] is None else '1' if f[k] else '0' for f in frames) for k in frames[0] if k!='i'}
    dump('after_gates.json',output)
    print('remaining audited',len(output),flush=True)
def visible_changes():
    # Replace by another real early-view recording of a different placement.
    # Verify what the tracker did, independently of the eight-step ODE endpoints.
    setup();m=model(dest,'ode_tracked.py')
    from overhead_track import StationaryCube
    refs={v['ep']:np.array(v['label']) for v in read(edge/'diag.json')['test']}
    recs=dict(rows('test'));cache={}
    for ep,rec in recs.items():
        items=[]
        for i in range(12):
            frame=rp.imaging(rec,i,'overhead')
            uv=m.aligner.det.positions(m.heatmap(frame))[0].cpu().numpy()
            _,view=m.registration.locate(frame,uv,m.aligner.par['matrix'])
            items.append((frame,uv,view))
        cache[ep]=items
    output=[]
    for a,b in read(edge/'archive.json')['pairs']:
        for old_id,new_id in ((a,b),(b,a)):
            tracker=StationaryCube();t=1_000_000_000;switch=None
            for frame,uv,view in cache[old_id]:
                pos,mode=tracker.update(t,frame,uv,view);t+=33_000_000
            old_error=None if pos is None else float(np.linalg.norm(pos-refs[old_id]))
            for i,(frame,uv,view) in enumerate(cache[new_id]):
                pos,mode=tracker.update(t,frame,uv,view);t+=33_000_000
                if switch is None and mode=='visible' and np.linalg.norm(pos-refs[new_id])<=12:switch=i+1
            output.append(dict(recipient=old_id,donor=new_id,distance_px=round(float(np.linalg.norm(refs[old_id]-refs[new_id])),2),
                initial_error_px=None if old_error is None else round(old_error,2),reacquired_frame=switch))
    dump('early_swaps.json',output);print('early reacquisition',sum(r['reacquired_frame'] is not None for r in output),len(output),flush=True)
def zero_motion():
    setup();m=model(dest,'ode_tracked.py')
    from wrist_xy import WristPosition
    saved=read(dest/'continuous_archive.json');result={}
    for ep,rec in rows('test'):
        case=saved[ep]
        if abs(case['natural'][1])>.001:continue
        phase=case['tracking']['phase']
        if phase=='hold_unobserved_xy':
            result[ep]=dict(reason='no_reliable_target',step_mm=0.);continue
        if not case['tracking']['wrist'] or phase!='approach':
            result[ep]=dict(reason='other_distance_change_under_resolution',step_mm=None);continue
        w=WristPosition(m.wrist_fits);j=rec['close']-18
        for i in range(max(0,j-11),j+1):
            row=rec['rows'][i]
            if not rp.valid(row):
                w=WristPosition(m.wrist_fits);continue
            q=rp.at(row,'feedback');t=m.aligner.flange(np.rad2deg(q[:6]));tip=t[:3,3]+t[:3,:3]@m.aligner.par['tip']
            over=rp.imaging(rec,i,'overhead')
            _,view=m.registration.locate(over,[0,0],m.aligner.par['matrix'])
            target=w.update(row['sample_monotonic_ns'],view,rp.imaging(rec,i,'wrist'),tip,t[:3,:3],m.relative.calibration)
        assert target is not None
        gap=float(np.linalg.norm(target-tip[:2]));err=m.wrist_fits[view]['uncertainty']
        step=min(float(m.aligner.par['cap']),float(m.aligner.par['gain'])*max(0.,gap-err))
        result[ep]=dict(reason='wrist_inside_error_allowance' if step<=.001 else 'other_distance_change_under_resolution',estimated_gap_mm=round(gap,3),step_mm=round(step,5))
    dump('zero_motion.json',result);print('zero-motion causes',len(result),flush=True)
if __name__=='__main__':
    if sys.argv[5]=='controls':controls()
    elif sys.argv[5]=='remaining':remaining()
    elif sys.argv[5]=='visible':visible_changes()
    elif sys.argv[5]=='zeros':zero_motion()
    else:natural(sys.argv[5])
