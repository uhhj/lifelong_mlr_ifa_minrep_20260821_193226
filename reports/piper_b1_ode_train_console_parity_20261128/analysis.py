"""Recompute paired metrics and public episode-level outcomes; preserve full-window diagnostics privately."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
base=Path(__file__).resolve().parent
prev=base.parent/'displacement_audit_1128'
def read(p):return json.loads(p.read_bytes())
def write(p,o,compact=False):
    p.write_bytes((json.dumps(o,ensure_ascii=False,allow_nan=False,indent=None if compact else 1)+'\n').encode())
def summary(vals):
    if not len(vals):return None
    a=np.asarray(vals,float)
    return dict(zip(('min','p50','p95','max'),np.round(np.percentile(a,[0,50,95,100]),7).tolist()))
def projection(e,v):
    n=np.linalg.norm(e[:2])
    return float(np.dot(e[:2],v[:2])/max(n,1e-12))
def xy(e):return float(np.linalg.norm(e[:2]))
def mark(e,v,phase):return projection(e,v)<0 if phase=='correct' else v[2]>=0
def phase_stats(rows):
    out={}
    for phase in ('correct','descend'):
        sub=[r for r in rows if r['phase']==phase]
        versions={}
        if sub:
            for name in sub[0]['v']:
                pairs=[(r,r['v'][name]) for r in sub if name in r['v']]
                score=[projection(r['expert'],v) if phase=='correct' else v[2] for r,v in pairs]
                versions[name]=dict(n=len(pairs),median=summary(score)['p50'],wrong=sum(bool(mark(r['expert'],v,phase)) for r,v in pairs))
        out[phase]=dict(windows=len(sub),episodes=len({r['ep'] for r in sub}),versions=versions)
    return out
def collect():
    results=[];ep_summary={};qc=[];overlap=[];train=read(base/'training_control.json')
    clock=read(base/'timing_checks.json');mix=read(base/'components.json');training=[]
    for e,ep in train['episodes'].items():
        for w in ep['windows']:
            training.append(dict(ep=e,t=w['t'],phase=w['phase'],expert=w['expert'],v=dict(expert=w['expert'],command=w['command'],**w['ends'])))
    for g in ('regression','archive'):
        pairs=read(base/('paired_'+g+'.json'));reference=read(prev/('windows_'+g+'.json'))
        assert pairs.keys()==reference.keys()
        for ep,src in pairs.items():
            if src['in_train']:overlap.append(ep)
            qc.append({**src['checks'],'filename_rounding_ns':src['source_file_stamp_ns']})
            raw={w['t']:w for w in reference[ep]['windows']};cases=[]
            assert len(src['windows'])==len(raw)
            for w in src['windows']:
                r=raw[w['t']];assert (r['grid'],r['phase'])==(w['grid'],w['phase'])
                row=dict(group=g,ep=ep,t=w['t'],grid=w['grid'],phase=w['phase'],expert=r['expert'],anchor=r['anchor'],v=dict(expert=r['expert'],command=r['command'],station=w['station'],**{k:q['end'] for k,q in w['variants'].items()}),
                          delta=w['variants'],timing=w['timing'])
                if w['grid']:
                    row['shifts']=clock[ep]['shifted'][str(w['t'])]
                    if w['phase']!='other':
                        row['mix']=next(m for m in mix[ep] if m['t']==w['t'])
                        row['v']['bf16']=row['mix']['bf16_end']
                cases.append(row)
            results+=cases
            primary=[r for r in cases if r['grid']]
            ep_summary[ep]=dict(group=g,window_count=len(cases),in_training=src['in_train'],checks=src['checks'],
                zero_endpoint_mm=summary([r['delta']['train_zero']['tip8'] for r in cases]),
                sampled_endpoint_mm=summary([r['delta']['component_a']['tip8'] for r in cases]),
                zero_reverse=sum(mark(r['expert'],r['v']['train_zero'],'correct') for r in primary if r['phase']=='correct'),
                zero_up=sum(int(r['v']['train_zero'][2]>=0) for r in primary if r['phase']=='descend'))
    all_grid=[r for r in results if r['grid']]
    ans=dict(windows=len(results),grid=len(all_grid),heldout_episodes=len(ep_summary),training_episodes=len(train['episodes']),
        training_rows=sum(z['rows'] for z in train['episodes'].values()),overlap=overlap,checkpoint=train['checkpoint'],source_sha=train['source_sha'],
        input_max={k:max(q[k] for q in qc) for k in qc[0]},training_checks=train['checks'],modes={},phase={},labels={},timing={},axes={},mix={},span={})
    for g,rs in [('all',results),('regression',[r for r in results if r['group']=='regression']),('archive',[r for r in results if r['group']=='archive'])]:
        grid=[r for r in rs if r['grid']];ans['phase'][g]=phase_stats(grid)
        ans['modes'][g]={k:{m:summary([r['delta'][k][m] for r in rs if k in r['delta']]) for m in ('tip_max','tip8','joint_deg','jaw_mm')} for k in ('train_zero','component_a','component_b','aug_a','aug_b')}
    ans['phase']['training']=phase_stats(training)
    for phase in ('correct','descend'):
        sub=[r for r in all_grid if r['phase']==phase];wr=[r for r in sub if mark(r['expert'],r['v']['station'],phase)]
        def signed(r,v):return projection(r['expert'],v) if phase=='correct' else -v[2]
        ans['labels'][phase]=dict(n=len(sub),wrong=len(wr),official_label_right_when_wrong=sum(signed(r,r['v']['command'])>0 for r in wr),
            all_shifts_right_when_wrong=sum(all(signed(r,v)>0 for v in r['shifts'].values()) and len(r['shifts'])==3 for r in wr),
            flips_zero_vs_station=sum(mark(r['expert'],r['v']['station'],phase)!=mark(r['expert'],r['v']['train_zero'],phase) for r in sub),
            flips_sample_a_vs_zero=sum(mark(r['expert'],r['v']['component_a'],phase)!=mark(r['expert'],r['v']['train_zero'],phase) for r in sub),
            flips_crop_a_vs_zero=sum(mark(r['expert'],r['v']['aug_a'],phase)!=mark(r['expert'],r['v']['train_zero'],phase) for r in sub),
            command_anchor_reverse=sum(projection(np.subtract(r['v']['command'],r['anchor']),np.subtract(r['v']['station'],r['anchor']))<0 for r in sub) if phase=='correct' else None)
        goodtime=[r for r in sub if r['timing']['skew']<16.67 and max(r['timing'][k] for k in ('target_joints_age','fb_joints_age'))<33.34]
        ans['labels'][phase].update(fresh=len(goodtime),fresh_wrong=sum(mark(r['expert'],r['v']['station'],phase) for r in goodtime))
        eligible=[r for r in wr if signed(r,r['v']['command'])>0]
        attrs=np.array([[(-projection(r['expert'],v) if phase=='correct' else v[2]) for v in r['mix']['axes']] for r in eligible])
        ans['axes'][phase]=dict(windows=len(eligible),mean_mm=np.round(attrs.mean(0),4).tolist(),median_mm=np.round(np.median(attrs,0),4).tolist(),
                                most_adverse=np.bincount(attrs.argmax(1),minlength=6).tolist())
        def prob(r):
            return sum(p for p,v in zip(r['mix']['weights'],r['mix']['ends']) if mark(r['expert'],v,phase))
        probs=[prob(r) for r in sub]
        ans['mix'][phase]=dict(n=len(sub),expected_wrong=round(sum(probs),3),expected_fraction=round(np.mean(probs),5),
            wrong_remain_expected=round(sum(prob(r) for r in wr),3),all_components_wrong=sum(all(mark(r['expert'],v,phase) for v in r['mix']['ends']) for r in sub),
            max_mode_check_mm=max(r['mix']['mode_recheck'] for r in sub),max_loss_error=max(r['mix']['loss_error'] for r in sub),
            bf16_tip_mm=summary([np.linalg.norm(np.subtract(r['v']['bf16'],r['v']['train_zero'])) for r in sub]))
        ans['span'][phase]={k:{which:sum(max(r['mix']['span'][k][0:6] if axes=='six' else r['mix']['span'][k])>1.00001 for r in group)
            for which,group,axes in [('all_any',sub,'seven'),('all_joints',sub,'six'),('wrong_any',wr,'seven'),('wrong_joints',wr,'six')]} for k in ('state','anchor','labels')}
    for group in ('all','regression','archive'):
        entries=[v for e,v in clock.items() if group=='all' or ep_summary[e]['group']==group]
        vals={k:sum((v['clock'][k] for v in entries),[]) for k in entries[0]['clock']}
        vals['sample_interval_ms']=sum((v['interval'] for v in entries),[])
        ans['timing'][group]={k:dict(**summary(a),n=len(a),negative=int(sum(v<0 for v in a))) for k,a in vals.items()}
    def combine(packets):
        counts=sum((Counter(r['counts']) for r in packets),Counter())
        return dict(counts=dict(counts),maximum={k:max(r['maximum'].get(k,0) for r in packets) for k in ('joints_rad','gripper_mm')})
    ans['packets']={g:combine([v['packet'] for e,v in clock.items() if g=='all' or ep_summary[e]['group']==g]) for g in ('all','regression','archive')}
    ans['packets']['training']=combine([e['packet'] for e in train['episodes'].values()])
    ds=read(base/'domain.json')['episodes']
    limits={e:{w['t']:w for w in vals} for e,vals in ds.items()}
    ans['component_tail']={}
    for g in ('regression','archive'):
        rs=[r for r in results if r['group']==g]
        tail=[r for r in rs if r['delta']['component_a']['tip8']>100]
        worst=max(rs,key=lambda r:r['delta']['component_a']['tip8'])
        def outside(rows):
            return {k:[sum(limits[r['ep']][r['t']][k][j]>1.0001 for r in rows) for j in range(7)] for k in ('agent_pos','action')}
        ans['component_tail'][g]=dict(n=len(rs),above50=sum(r['delta']['component_a']['tip8']>50 for r in rs),above100=len(tail),
             outside_all=outside(rs),outside_above100=outside(tail),worst=dict(id=worst['ep'],frame=worst['t'],phase=worst['phase'],tip_mm=worst['delta']['component_a']['tip8'],joint_deg=worst['delta']['component_a']['joint_deg'],jaw_mm=worst['delta']['component_a']['jaw_mm']))
    return ans,ep_summary,results
