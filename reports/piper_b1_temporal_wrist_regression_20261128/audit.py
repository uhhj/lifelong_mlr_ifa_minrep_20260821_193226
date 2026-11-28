"""Verify the unchanged STOP contract, negative field controls, and registration ablations offline."""
from score import *
def audit():
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    field_model=rp.models()[0]
    candidate=new_model()
    import verify,guard_case,runner
    key='B1-ode50-wrist-20261127'
    p=read(root/'candidate.json')
    assert p['end_mode']==runner.PROFILES[key]['end_mode']=='stop'
    try:runner.uses_fixed_grip(key,p)
    except ValueError:pass
    else:raise AssertionError('Unexpected registry acceptance')
    names=['preview_only','return_pending','unknown_mode','confirm_only','outside_pose','usb_unready','success','failed','stop','close','deadline','release','heartbeat','stale_prediction','no_stop_ack']
    result=dict(cases=[guard_case.exercise(n,key) for n in names],heartbeat_ns=runner.SAFETY_NS,freshness_ns=runner.FRESH_NS,
                field=verify.field([field_model,candidate]))
    diag=read(root/'diag.json')
    gt=np.asarray(read(work/'offline_diagnostics/b1_vision_contrasts_20261124/final.json')['geometry']['table_map'])
    actual=np.asarray(candidate.aligner.par['matrix'])
    older=np.load(root/'private_scene_map.npz')['raw_matrix']
    errors=[]
    for r in diag['test']:
        uv=np.r_[r['late'],1];hand=np.r_[r['label'],1];dest=np.array(r['goal'])
        errors.append(dict(ep=r['ep'],**r['errors'],old_px_new_map_mm=float(np.linalg.norm(uv@older-dest)),
              late_static_px=float(np.linalg.norm(uv[:2]-hand[:2])),early_new_map_mm=float(np.linalg.norm(np.r_[r['early'],1]@older-dest)),
              runtime_old_mm=float(np.linalg.norm(uv@actual-dest)),manual_runtime_mm=float(np.linalg.norm(hand@actual-dest)),
              delta_xy=(hand@actual-dest).tolist(),contact_ref_difference_mm=float(np.linalg.norm(dest-r['low_goal']))))
    result.update(registration=errors,archived_map_equal_to_runtime=bool(np.allclose(gt,actual)),n_cal=len(diag['train']),
                  n_disjoint=len({r['ep'] for r in diag['train']}&{r['ep'] for r in diag['test']}),
                  training_holdout_overlap=len(set(read(prev/'old_manifest.json')['selected'])&set(read(prev/'archive.json')['episodes'])))
    dump('audit.json',result)
    print('audited',len(result['field']),len(names),'controls',flush=True)
def missed_windows():
    from unittest.mock import patch
    from collections import Counter
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    rp.models()
    m=new_model()
    from b1_placement import B1Placement
    from b1_alignment_gate import OVERHEAD_TOLERANCE_MM
    from assess import reviewed
    refs=reviewed(work,work/'offline_diagnostics/b1_vision_contrasts_20261124')
    details={}
    for group in ('regression','archive'):
        oldrows=read(root/(group+'.json'))['episodes']
        if group=='archive':records=dict(rows('test'))
        else:
            records={r['episode']:dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in refs if r['episode'] in oldrows}
        for ep,row in oldrows.items():
            if row['close_events'][1]:continue
            rec=records[ep];evidence=Counter();original=B1Placement.next_goal
            def log(gate,stamp,intent,xyz,width,*,vision):
                goal=original(gate,stamp,intent,xyz,width,vision=vision)
                if intent:
                    evidence['intent']+=1
                    near=vision.get('view') is not None and vision['distance_mm']<=OVERHEAD_TOLERANCE_MM
                    evidence['near']+=int(near)
                    evidence['edge_near']+=int(near and vision.get('edge_centered',False))
                    evidence['aligned']+=int(vision['aligned'])
                return goal
            with patch.object(B1Placement,'next_goal',log):
                assert rp.close_event(m,rec) is None
            reason=next((reason for reason,k in [('no_intent','intent'),('no_coarse_agreement','near'),('no_wrist_confirmation','aligned')] if not evidence[k]),'insufficient_consecutive_frames')
            details[ep]=dict(group=group,reason=reason,frames=dict(evidence))
    dump('misses.json',details)
    print('missing windows',len(details),Counter(x['reason'] for x in details.values()),flush=True)
def endpoints():
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    rp.models();m=new_model()
    truths={r['ep']:np.array(r['goal']) for r in read(root/'diag.json')['test']}
    results=[]
    for ep,r in rows('test'):
        end,start=rp.endpoint(m,r,r);goal=truths[ep]
        coarse=np.r_[m.visual['block_uv'],1]@m.aligner.par['matrix']
        def dist(a,b):return float(np.linalg.norm(a-b))
        results.append(dict(ep=ep,active=m.details['applied'],initial=dist(start,goal),final=dist(end,goal),step=dist(end,start),
            coarse_error=dist(coarse,goal),coarse_advance=dist(start,coarse)-dist(end,coarse),
            wrist_error=None if m.relative.target is None else dist(m.relative.target,goal)))
    dump('endpoints.json',results)
    print('archived endpoints',len(results),flush=True)
if __name__=='__main__':
    if len(sys.argv)>4 and sys.argv[4]=='misses':missed_windows()
    elif len(sys.argv)>4 and sys.argv[4]=='endpoints':endpoints()
    else:audit()
