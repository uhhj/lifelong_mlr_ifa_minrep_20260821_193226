"""Run the two frozen B1 regressions with the eight-frame swap and expert close window."""
from basis import *
import hashlib,importlib.util

def dump(name,obj):(dest/name).write_text(json.dumps(obj,allow_nan=False,indent=2))
def new_model():return model(dest,'ode_tracked.py')

def evaluate(group):
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    oldresults=read(edge/(group+'.json'))
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    sys.path.insert(0,str(work/'offline_diagnostics/b1_vision_contrasts_20261124'))
    from assess import reviewed
    gt=np.asarray(read(prior/'final.json')['geometry']['table_map'])
    baseline=setup()
    m=new_model()
    if group=='regression':
        refs=reviewed(work,prior)
        recs={r['episode']:dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in refs if r['episode'] in oldresults['episodes']}
        goals={r['episode']:np.r_[r['uv'],1.]@gt for r in refs if r['episode'] in recs}
    else:
        recs=dict(rows('test'));goals={r['ep']:np.array(r['goal']) for r in read(edge/'diag.json')['test']}
    assert recs.keys()==oldresults['episodes'].keys()
    out=dict(group=group,sha=read(dest/'identity.json')['profile'],pairs=oldresults['pairs'],episodes={},directions=[])
    ends={};tiles={};checks=[]
    for n,(ep,r) in enumerate(recs.items()):
        oldrow=oldresults['episodes'][ep]
        assert oldrow['alignment_sha256']==rp.sha(r['folder']/'alignment.jsonl') and r['close']==oldrow['close_frame']
        assert rp.eligible(r)
        end,start=rp.endpoint(m,r,r)
        ends[ep]=end
        natural=float(np.linalg.norm(start-goals[ep])-np.linalg.norm(end-goals[ep]))
        trace=m.trace['alignment']
        assert trace['stamp']==r['rows'][r['close']-18]['sample_monotonic_ns']
        cand_event=rp.close_event(m,r)
        if cand_event:
            fr=cand_event['frame']
            cand_event.update(mode=m.visual.get('edge_mode'),edge=bool(m.visual.get('edge_centered')),metric=bool(m.visual.get('fresh_metric_center')))
            token=hashlib.sha256((ep+str(fr)).encode()).hexdigest()[:8]
            cand_event['token']=token
            tiles[token]=rp.imaging(r,fr,'wrist')
        def geometry(event):
            if event is None:return None
            poses=[]
            for k in (event['frame'],r['close']):
                q=rp.at(r['rows'][k],'feedback');f=m.aligner.flange(np.rad2deg(q[:6]))
                poses.append(f[:3,3]+f[:3,:3]@m.aligner.par['tip'])
            delta=poses[0]-poses[1]
            return dict(event,delta_z_to_expert_mm=round(float(delta[2]),2),delta_xy_to_expert_mm=round(float(np.linalg.norm(delta[:2])),2))
        out['episodes'][ep]=dict(region=oldrow['region'],alignment_sha256=oldrow['alignment_sha256'],close_frame=r['close'],
            active=trace['applied'],view=trace['view'],track=trace['track_mode'],raw_uv=trace['detected_uv'],used_uv=trace['tracker_uv'],natural_mm=[oldrow['natural_mm'][1],round(natural,4)],
            close_events=[geometry(oldrow['close_events'][1]),geometry(cand_event)])
        if n in (0,len(recs)-1):
            e0,s0=rp.endpoint(baseline,r,r)
            checks.append(abs(float(np.linalg.norm(s0-goals[ep])-np.linalg.norm(e0-goals[ep]))-oldrow['natural_mm'][1]))
            old_event=rp.close_event(baseline,r)
            expected=oldrow['close_events'][1]
            assert checks[-1]<.01 and (old_event or {}).get('frame')==(expected or {}).get('frame')
        if (n+1)%8==0:print(group,n+1,'of',len(recs),flush=True)
    for item in oldresults['directions']:
        ep,other=item['recipient'],item['donor']
        axis=goals[other]-goals[ep];axis=axis/np.linalg.norm(axis)
        end,start=rp.endpoint(m,recs[ep],recs[other])
        response=float((end-ends[ep])@axis)
        out['directions'].append(dict(recipient=ep,donor=other,region=item['region'],correction_mm=[item['correction_mm'][1],round(response,5)]))
    out['baseline_recalculation_mm']=checks
    dump(group+'.json',out)
    for i in range(0,len(tiles),10):
        sheet=[]
        for token in sorted(tiles)[i:i+10]:
            im=cv2.resize(tiles[token],(320,240));tile=cv2.copyMakeBorder(im,20,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,token,(3,15),0,.5,(255,255,255),1);sheet.append(tile)
        if len(sheet)%2:sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(str(dest/f'{group}{i}.jpg'),np.vstack([np.hstack(sheet[k:k+2]) for k in range(0,len(sheet),2)]))
    print('complete',group,len(tiles),'event frames',flush=True)
if __name__=='__main__':evaluate(sys.argv[5])
