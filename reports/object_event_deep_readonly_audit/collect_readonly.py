"""Existing metadata/HDF5 only. No torch, model, optimizer, env, or selection execution."""
import json,hashlib,statistics,collections,os
from pathlib import Path
import numpy as np
import h5py

OUT=Path('/tmp/event_deep_audit_20261017.json')
hashes={};hdfstat={}
def read(p):
    p=Path(p); b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def hist(xs):return dict(collections.Counter(xs))
def stats(xs):
    xs=list(xs)
    return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,max=max(xs) if xs else None)
def summary_positions(ids,windows):
    pairs={(windows[x]['demo_id'],windows[x]['target_step']) for x in ids}
    adjacent=sum((d,t-1) in pairs for d,t in pairs)
    return dict(unique_demo_targets=len(pairs),adjacent_target_pairs=adjacent,padded=sum(not all(windows[x]['valid_mask']) for x in ids))

allrows=[];position_rows=[];epoch_rows=[]
for seed in range(40,45):
    if seed<42:
        base=Path('/data/mlr_ifa/experiments/Seeds40_41_three_arm_20261016/seed40' if seed==40 else '/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41')
        ec,rc=base/'ec',base/'rc'
    else:
        ec=Path('/data/mlr_ifa/experiments/Event_Coreset_OBJECT_seed%d_20261015'%seed)
        rc=Path('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed%d_20261013'%seed)
    binding=read(ec/'WINDOW_BINDING.json'); stored={'C':{},'EC':{}}
    for task in range(6,10):
        a=read(ec/'event_audit'/('task%d.json'%task))
        parent=read(rc/'coreset/storage'/('task%d.json'%task))
        windows={r['sample_id']:r for r in binding['windows'] if r['task']==task}
        c=[Path(r['source']).name for r in parent['selected']];e=a['selected_ids']
        assert len(c)==len(e)==a['q']; cs,es=set(c),set(e)
        added=es-cs;removed=cs-es;eventids=set(a['event_ids'])
        assert added<=eventids and not removed&eventids
        stored['C'][task]=c;stored['EC'][task]=e
        events=[x for x in a['events'] if x['status']=='confirmed']
        bydemo=collections.defaultdict(list)
        for i,x in enumerate(events):bydemo[x['demo_id']].append((i,x))
        covers={x:[] for x in windows}
        for x,w in windows.items():
            covers[x]=[i for i,v in bydemo[w['demo_id']] if v['b']<=w['target_step']<=v['e']]
        cc=collections.Counter(i for x in c for i in covers[x]);ee=collections.Counter(i for x in e for i in covers[x])
        candidate_counts=collections.Counter(i for x in windows for i in covers[x])
        per_event=[dict(**v,interval_length=v['e']-v['b']+1,candidate_windows=candidate_counts[i],c_windows=cc[i],ec_windows=ee[i]) for i,v in enumerate(events)]
        def type_summary(ids):
            return hist('+'.join(sorted({events[i]['type'] for i in covers[x]})) or 'regular' for x in ids)
        before_after={}
        for name,ids in [('added',added),('removed',removed)]:
            near_before=near_after=near_either=0
            for x in ids:
                w=windows[x];t=w['target_step'];vs=[v for _,v in bydemo[w['demo_id']]]
                bef=any(1<=v['b']-t<=7 for v in vs);aft=any(1<=t-v['e']<=7 for v in vs)
                near_before+=bef;near_after+=aft;near_either+=bef or aft
            before_after[name]={'within7_before_confirmed_interval':near_before,'within7_after_confirmed_interval':near_after,'within7_either':near_either}
        hdf=Path(a['adapter']['hdf5']);st=hdf.stat();hdfstat[str(hdf)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
        with h5py.File(hdf,'r') as hf:
            actions={d:hf['data'][d]['actions'][:] for d in {w['demo_id'] for w in windows.values()}}
        def action_summary(ids):
            acts=np.array([actions[windows[x]['demo_id']][windows[x]['target_step']] for x in ids])
            if not len(acts):return {'n':0}
            return dict(n=len(acts),gripper_close_fraction=float(np.mean(acts[:,-1]>0)),translation_rms_mean=float(np.sqrt((acts[:,:3]**2).mean(axis=1)).mean()),rotation_rms_mean=float(np.sqrt((acts[:,3:6]**2).mean(axis=1)).mean()))
        lengths=[v['e']-v['b']+1 for v in events];median_len=statistics.median(lengths)
        adds_in_covered=sum(any(cc[i]>0 for i in covers[x]) for x in added)
        row=dict(seed=seed,task=task,q=a['q'],added=len(added),removed=len(removed),event_windows_candidate=len(eventids),
            c_event_windows=len(cs&eventids),ec_event_windows=len(es&eventids),
            confirmed_events=len(events),c_covered_events=sum(cc[i]>0 for i in range(len(events))),ec_covered_events=sum(ee[i]>0 for i in range(len(events))),
            newly_covered_events=sum(cc[i]==0 and ee[i]>0 for i in range(len(events))),lost_covered_events=sum(cc[i]>0 and ee[i]==0 for i in range(len(events))),
            added_windows_with_previously_covered_event=adds_in_covered,
            added_windows_in_longer_than_median_event=sum(any(lengths[i]>median_len for i in covers[x]) for x in added),
            lengths=stats(lengths),statuses=hist(v['type']+':'+v['status']+':'+str(v['reason']) for v in a['events']),
            added_type=type_summary(added),removed_type=type_summary(removed),c_types=type_summary(c),ec_types=type_summary(e),
            positions={name:summary_positions(ids,windows) for name,ids in [('C',c),('EC',e),('added',added),('removed',removed)]},
            action_stats={name:action_summary(ids) for name,ids in [('C',c),('EC',e),('added',added),('removed',removed)]},
            near_event_boundary=before_after,per_event=per_event)
        allrows.append(row)
    for stage in (7,8,9):
        lists={m:sorted('t%d__%s'%(t,n) for t in range(6,stage) for n in stored[m][t]) for m in ('C','EC')}
        ca,ea=lists['C'],lists['EC'];assert len(ca)==len(ea)
        cmap={n:i for i,n in enumerate(ca)};emap={n:i for i,n in enumerate(ea)};shared=set(ca)&set(ea)
        taskn=next(r['N'] for r in [read(ec/'event_audit'/('task%d.json'%stage))])
        changed=sum(x!=y for x,y in zip(ca,ea))
        position_rows.append(dict(seed=seed,stage=stage,buffer_slots=len(ca),full_dataset_size=len(ca)+taskn,
            replaced_members=len(set(ca)-set(ea)),same_sorted_slot_identity=sum(x==y for x,y in zip(ca,ea)),
            changed_sorted_slots=changed,full_dataset_changed_slot_percent=changed/(len(ca)+taskn)*100,
            common_members_moved=sum(cmap[n]!=emap[n] for n in shared),
            task_slot_labels_unchanged=all(x.split('__')[0]==y.split('__')[0] for x,y in zip(ca,ea))))
    ce=read(rc/'coreset/epochs.json');ee=read(ec/'event_coreset/epochs.json')
    for t in range(6,10):
        x=[r for r in ce if r['task']==t];y=[r for r in ee if r['task']==t]
        epoch_rows.append(dict(seed=seed,task=t,epochs=len(x),loss_logs_exactly_equal=all(a['loss']==b['loss'] for a,b in zip(x,y)),first_different_epoch=next((a['epoch'] for a,b in zip(x,y) if a['loss']!=b['loss']),None)))

assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert all(Path(p).stat().st_size==s['size'] and Path(p).stat().st_mtime_ns==s['mtime_ns'] for p,s in hdfstat.items())
result=dict(status='PASS_READONLY',metadata_sha256=hashes,hdf5_stat_before_after=hdfstat,hdf5_check='size_and_mtime_only_not_new_content_hash',rows=allrows,sorted_slot_analysis=position_rows,epoch_log_comparison=epoch_rows)
OUT.write_text(json.dumps(result,indent=2));print(json.dumps({'status':'PASS_READONLY','metadata_files':len(hashes),'rows':len(allrows),'output':str(OUT)}))
