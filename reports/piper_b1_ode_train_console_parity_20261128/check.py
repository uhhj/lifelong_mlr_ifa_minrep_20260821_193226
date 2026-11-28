"""Controls using the genuine persisted training cache and the source command packets."""
from inputs import *
from parity import evaluate,end_vec
from collections import Counter
import struct
def packets(rows):
    totals=Counter();top={}
    seen=set()
    for r in rows:
        k=r['sample_monotonic_ns']
        if k in seen:continue
        seen.add(k)
        c=r['target'];h=c.get('frame_hex')
        if not isinstance(h,list) or len(h)!=4:totals['not_stored']+=1;continue
        joints=[v for raw in h[:3] for v in struct.unpack('>ii',bytes.fromhex(raw))]
        width=struct.unpack('>i',bytes.fromhex(h[3])[:4])[0]*.001
        err=float(abs(np.array(joints)*np.pi/180000-np.array(c['joints_rad'])).max())
        werr=abs(abs(width)-c['gripper_mm'])
        maxes(top,joints_rad=err,gripper_mm=werr);totals['decoded']+=1
        totals['joint_mismatch']+=int(err>1e-7);totals['jaw_mismatch']+=int(werr>1e-4)
    return dict(counts=totals,maximum=top)
def shifts(seq,t,origin,station):
    result={}
    for k in (-1,0,1):
        i=t+7+k
        if not seq[i].get('target'):continue
        result[str(k)]=end_vec(xyz(station,np.asarray(rp.at(seq[i],'target')))-origin)
    return result
def clockrows(seq,indices):
    result={}
    for i in sorted(indices):
        r=seq[i];q=r['target'];f=r['feedback'];s=r['sample_monotonic_ns']
        head,wrist=[r['cameras'][role]['monotonic_ns'] for role in roles]
        d={'camera_skew':abs(head-wrist),'feedback_age':s-f['sample_monotonic_ns'],'target_age':s-q['sample_monotonic_ns'],
           'six_joint_target_oldest':s-min(q['joint_bus_monotonic_ns']), 'six_joint_feedback_oldest':s-min(f['component_monotonic_ns'][:3]),
           'target_span':max(q['joint_bus_monotonic_ns'])-min(q['joint_bus_monotonic_ns']),
           'head_minus_target':head-q['sample_monotonic_ns'],'wrist_minus_target':wrist-q['sample_monotonic_ns']}
        for key,v in d.items():result.setdefault(key,[]).append(v/1e6)
    return result
@torch.inference_mode()
def run():
    train,m,stats=init()
    full=TrainingFrames(cache,'B1');arrays,episodes=read_arrays(cache,'B1')
    errors={};control={};meta=read(cache/'metadata.json');select=read(root/'selection.json')
    selected={r['id']:r for r in select['ranking'] if r['id'] in select['selected_ids']}
    assert set(selected)=={r['id'] for r in episodes}
    for key in ('action','agent_pos'):
        for field in ('min','max','mean','std'):maxes(errors,**{key+'_'+field:np.max(abs(np.array(stats[key][field])-full.stats[key][field]))})
    ts=np.load(cache/'stamps_ns.npy',mmap_mode='r')
    for ep in episodes:
        e=ep['id'];r=selected[e];d=B/r['source'];l,h=ep['start'],ep['stop'];assert rp.sha(d/'alignment.jsonl')==r['alignment_sha256']
        seq=[json.loads(x) for x in (d/'alignment.jsonl').read_bytes().splitlines()][r['first']:]
        assert len(seq)==h-l
        for role,key in zip(roles,('head_camera','wrist_camera')):
            paths=files(d,role)[r['first']:]
            assert len(paths)==len(seq)
            pixerr=0
            for i,p in enumerate(paths):
                actual=train_rgb(p);stored=arrays[key][l+i]
                if not np.array_equal(actual,stored):pixerr=max(pixerr,int(abs(actual.astype('int16')-stored).max()))
            maxes(errors,**{key+'_stored_u8':pixerr})
        fb=vec(seq,'feedback');cmd=vec(seq,'target')
        assert np.array_equal(ts[l:h],[x['sample_monotonic_ns'] for x in seq])
        for k,w in [('state',fb),('command',cmd),('action',np.vstack((fb[:1],cmd[:-1])))]:
            maxes(errors,**{k+'_cache':np.max(abs(arrays[k][l:h]-w.astype(np.float32)))})
        c=next((i for i in range(1,len(seq)) if cmd[i-1,6]>55>=cmd[i,6]),None)
        if c is None or not any(np.all(fb[i:i+3,6]<=49) for i in range(max(0,c-6),len(seq)-2)):
            control[e]=dict(windows=[],rows=h-l,packet=packets(seq),reason='no comparable sustained grasp reference')
            continue
        _,stamp,_,_,x,xc,contact,goal=recorded(dict(rows=seq,close=c),m)
        windows=[]
        for t in range(c-8,6,-8):
            if np.min(cmd[t:t+8,6])<=55 or np.max(np.diff(stamp[t-7:t+9]))>70_000_000:continue
            exp=x[t+8]-x[t];phase=classify(x[t],x[t+8],goal,min(fb[t,6],fb[t+8,6]))
            if phase=='other':continue
            batch=full.batch(torch.tensor([l+t],device='cuda'))
            preds=evaluate(train,batch,False)
            windows.append(dict(t=t,phase=phase,expert=end_vec(exp),command=end_vec(xc[t+7]-x[t]),
                   anchor=end_vec(xc[t-1]-x[t]),ends={k:end_vec(output_xyz(m,v)[-1]-x[t]) for k,v in preds.items()}))
        control[e]=dict(windows=windows,rows=h-l,packet=packets(seq))
        if len(control)%10==0:print('real training controls',len(control),flush=True)
    spec={}
    for p in [root/'selection.json',root/'lerobot_io.py',root/'select_b1.py',code/'train_task.py',code/'verify_task.py',code/'ode_policy_adapter.py',live/'ode_wrist_alignment.py',live/'ode_live.py']:
        spec[p.name]=rp.sha(p)
    write('training_control.json',dict(episodes=control,checks=errors,source_sha=spec,checkpoint=read(last/'identity_check.json')['ode']))
    result={}
    for group in ('regression','archive'):
        prevdata=read(last/('windows_'+group+'.json'))
        for ep,rec in subjects(group):
            seq=rec['rows'];ws=prevdata[ep]['windows'];idx={i for w in ws for i in range(w['t']-7,w['t']+8)}
            grid=[w['t'] for w in ws if w['grid']]
            result[ep]=dict(packet=packets([seq[i] for i in idx]),clock=clockrows(seq,idx),
                shifted={t:shifts(seq,t,xyz(m,np.asarray(rp.at(seq[t],'feedback'))),m) for t in grid},
                interval=[(seq[i]['sample_monotonic_ns']-seq[i-1]['sample_monotonic_ns'])/1e6 for i in sorted(idx) if i-1 in idx])
    write('timing_checks.json',result)
if __name__=='__main__':
    if len(sys.argv)>7 and sys.argv[7]=='components':
        import mechanism
        mechanism.run()
    elif len(sys.argv)>7 and sys.argv[7]=='domain':
        import domain
        domain.run()
    else:run()
