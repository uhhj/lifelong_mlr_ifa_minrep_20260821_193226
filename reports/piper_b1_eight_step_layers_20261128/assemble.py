"""Recompute the review tables from private per-window displacement outcomes."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
from measures import *
here=Path(__file__).resolve().parent
previous=here.parent/'occlusion_final_1128'
def read(f):return json.loads(f.read_bytes())
def write(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False)+'\n',encoding='utf-8')
def gather():
    byep={};groups={};episodes={};meta={};rep={};hashes={}
    for group,want in [('regression',32),('archive',20)]:
        wp=here/('windows_'+group+'.json');sp=here/('rolling_'+group+'.json')
        d=read(wp);s=read(sp);prior=read(previous/(group+'.json'))['episodes']
        micro_file=here/('micro_'+group+'.json')
        micro=read(micro_file);hashes[micro_file.name]=hashlib.sha256(micro_file.read_bytes()).hexdigest()
        hashes[wp.name]=hashlib.sha256(wp.read_bytes()).hexdigest();hashes[sp.name]=hashlib.sha256(sp.read_bytes()).hexdigest()
        assert len(d)==want and d.keys()==s.keys()==prior.keys()
        groups[group]=[]
        for ep,episode in d.items():
            entries=episode['windows'];chosen={}
            assert ep not in byep and len({r['t'] for r in entries})==len(entries)
            assert episode['input_sha']==prior[ep]['alignment_sha256'] and episode['anchor_check_mm']<.003
            for r in entries:
                r['ep']=ep;r['region']=prior[ep]['region'];r['group']=group
                roll=s[ep].get(str(r['t']))
                assert bool(roll)==(r['phase']!='other' or r['t']==episode['close']-18)
                extra=micro.get(ep,{}).get(str(r['t']))
                r['micro']=bool(extra)
                if extra:
                    assert 1<=size(r['expert'])<3 and r['grid'] and r['phase']=='other'
                    if roll:assert roll==extra
                    roll=extra
                if roll:
                    r['rolling']=roll
                    assert set(roll)==set(VERSIONS) and all(z['initial_check_mm']<.003 and ((z['final'] is None)==bool(z['guard'])) for z in roll.values())
                assert (r['d0']-r['d8']>=1.999 and size(r['expert'])>=2.999) or r['phase']!='correct'
                assert (-r['expert'][2]>=2.999 and size(r['expert'])<=2.001) or r['phase']!='descend'
            for phase in PHASES[:2]:
                chosen[phase]=median_rep(entries,phase,episode['close']-18)
            if not any(chosen.values()):chosen['other']=median_rep(entries,'other',episode['close']-18)
            rep[ep]={k:r for k,r in chosen.items() if r}
            byep[ep]=entries;groups[group]+=entries
            meta[ep]=dict(group=group,region=prior[ep]['region'],close=episode['close'],contact=episode['contact'],
                 input_sha=episode['input_sha'],reference_gap_mm=episode['reference_gap_mm'],anchor_delta=episode['anchor_check_mm'],
                 grid=dict(Counter(r['phase'] for r in entries if r['grid'])),all=dict(Counter(r['phase'] for r in entries)),excluded=episode['excluded'])
    return byep,groups,meta,rep,hashes
def fold(rows):
    out={}
    for phase in PHASES[:2]:
        phase_rows=[r for r in rows if r['phase']==phase]
        out[phase]={n:describe(phase_rows,n) for n in VERSIONS}
    return out
def summary(groups,meta,reps,hashes):
    sets={**groups,'both':sum(groups.values(),[])}
    out=dict(file_sha256=hashes,counts={},primary={},small={},paired={},agreement={},representatives={},regions={},sensitivity={})
    for g,rows in sets.items():
        grid=[r for r in rows if r['grid']];m=[v for v in meta.values() if g=='both' or v['group']==g]
        out['counts'][g]=dict(episodes=len(m),windows=len(rows),grid=len(grid),grid_phase=Counter(r['phase'] for r in grid),
            all_phase=Counter(r['phase'] for r in rows),max_anchor_error=max(x['anchor_delta'] for x in m),
            ref_median=med(x['reference_gap_mm'] for x in m),ref_max=max(x['reference_gap_mm'] for x in m),
            excluded=sum(sum(x['excluded'].values()) for x in m), grid_episodes={p:len({r['ep'] for r in grid if r['phase']==p}) for p in PHASES})
        out['primary'][g]=fold(grid)
        out['small'][g]=fold([r for r in grid if r['phase']=='correct' and size(r['expert'])<=8])['correct']
        out['representatives'][g]=fold([r for rr in reps.values() for r in rr.values() if g=='both' or r['group']==g])
        out['paired'][g]={n:{p:describe([r for r in grid if r['phase']==p and r.get('rolling',{}).get(n,{}).get('final') is not None],n)
             for p in PHASES[:2]} for n in VERSIONS}
        out['agreement'][g]=fold([r for r in grid if r['image_phase']==r['phase']])
        out['sensitivity'][g]={p:{k:dict(Counter(r[k] for r in grid if r['phase']==p))
                                  for k in ('feedback7_phase','image_phase','phase_lo','phase_hi')} for p in PHASES[:2]}
        out['regions'][g]={region:{p:{n:compact_stats([r for r in grid if r['region']==region and r['phase']==p],n) for n in VERSIONS} for p in PHASES[:2]}
               for region in sorted({r['region'] for r in rows})}
    zero=read(previous/'zero_motion.json');prior_rows=sets['archive']
    out['previous_zero']=[dict(id=r['ep'],t=r['t'],phase=r['phase'],image_phase=r['image_phase'],prior_reason=zero[r['ep']]['reason'],
        distance_before=r['d0'],distance_after=r['d8'],expert_xy=round(size(r['expert']),3),expert_z=r['expert'][2],
        raw_p=round(projection(r['expert'],r['raw']),3),candidate_plan_xy=round(size(r['candidate']['plan']),3))
        for r in prior_rows if r['ep'] in zero and r['t']==meta[r['ep']]['close']-18]
    assert len(out['previous_zero'])==len(zero)==12
    out['micro']=[dict(id=r['ep'],t=r['t'],expert_mm=round(size(r['expert']),4),raw_along=round(projection(r['expert'],r['raw']),4),
        versions={n:dict(plan_along=round(projection(r['expert'],r[n]['plan']),4),final_along=(round(projection(r['expert'],r['rolling'][n]['final']),4) if r['rolling'][n]['final'] is not None else None))
        for n in VERSIONS}) for r in sets['both'] if r['micro']]
    return out
def compact_stats(rows,n):
    s=distances(rows,n)
    return dict(count=len(rows),episodes=len({r['ep'] for r in rows}),stages={k:[s[k]['n'],s[k]['xy'],s[k]['projection'],s[k]['z'],s[k]['passes']]
        for k in ('expert','raw','plan','final')})
