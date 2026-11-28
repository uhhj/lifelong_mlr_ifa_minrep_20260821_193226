"""Compute base-frame displacement metrics, using the expert to select and weight windows."""
import math
from collections import Counter
from statistics import median

PHASES=('correct','descend','other')
VERSIONS=('field','candidate')
def size(v):return math.hypot(v[0],v[1])
def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def projection(expert,v):return dot(expert,v)/max(size(expert),1e-12)
def sub(a,b):return [x-y for x,y in zip(a,b)]
def diffxy(a,b):return size(sub(a,b))
def med(xs):
    vals=[x for x in xs if x is not None]
    return round(median(vals),4) if vals else None
def ok(e,v):
    if v is None:return None
    p=projection(e,v)
    return size(v)>=.5 and p>=.5*size(e) and p>=.5*size(v)
def heading(v):
    return None if v is None or size(v)<.5 else round(math.degrees(math.atan2(v[1],v[0])),2)
def attribution(r,n):
    if r['phase']!='correct':return ''
    e=r['expert'];x=values(r,n)
    raw,plan,req,final=[ok(e,x[k]) for k in ('raw','plan','requested','final')]
    notes=['ODE 已达标' if raw else 'ODE 偏弱或偏向']
    if raw!=plan:notes.append('视觉计划补回' if plan else '视觉计划丢失')
    if final is None:notes.append('回放中断')
    else:
        if plan!=req:notes.append('滚动末步请求补回' if req else '滚动末步请求丢失')
        if req!=final:notes.append('末次限幅后达标' if final else '末次限幅后丢失')
    return ' → '.join(notes)
def angle(e,v):
    if size(e)<.5 or size(v)<.5:return None
    return math.degrees(math.acos(max(-1,min(1,dot(e,v)/size(e)/size(v)))))
def median_rep(rows,phase,anchor):
    arr=[r for r in rows if r['phase']==phase]
    if not arr:return None
    if phase=='other':return min(arr,key=lambda r:abs(r['t']-anchor))
    grid=[r for r in arr if r['grid']] or arr
    val=lambda r:size(r['expert']) if phase=='correct' else -r['expert'][2]
    center=median(val(r) for r in grid)
    return min(grid,key=lambda r:(abs(val(r)-center),r['t']))
def values(r,n):
    roll=r.get('rolling',{}).get(n,{})
    return dict(expert=r['expert'],command=r['command'],raw=r['raw'],plan=r[n]['plan'],bound=r[n]['bound'],
                requested=roll.get('requested'),final=roll.get('final'))
def distances(rows,n):
    out={}
    for k in ('expert','command','raw','plan','bound','requested','final'):
        pairs=[(r,values(r,n)[k]) for r in rows if values(r,n)[k] is not None]
        out[k]=dict(n=len(pairs),xy=med(size(v) for _,v in pairs),z=med(v[2] for _,v in pairs),
             projection=med(projection(r['expert'],v) for r,v in pairs),xy_error=med(diffxy(r['expert'],v) for r,v in pairs),
             angle=med(angle(r['expert'],v) for r,v in pairs),negative=sum(projection(r['expert'],v)<0 for r,v in pairs),
             negligible=sum(size(v)<.5 for _,v in pairs),above3=sum(size(v)>3 for _,v in pairs),
             z_up_or_level=sum(v[2]>=0 for _,v in pairs),z_descend3=sum(v[2]<=-3 for _,v in pairs),
             along_above_twice=sum(projection(r['expert'],v)>2*size(r['expert']) for r,v in pairs),
             amplitude_above_twice=sum(size(v)>2*size(r['expert']) for r,v in pairs),
             passes=sum(ok(r['expert'],v) for r,v in pairs))
    return out
def transitions(rows,n):
    counts=Counter();sources=Counter()
    for r in rows:
        vs=values(r,n);e=r['expert'];a,p,f=[ok(e,vs[k]) for k in ('raw','plan','final')]
        counts['raw_good' if a else 'raw_weak']+=1
        counts[('raw_good' if a else 'raw_weak')+('_plan_good' if p else '_plan_weak')]+=1
        if a and not p:sources[r[n]['source']]+=1
        if f is None:counts['guard_stop']+=1;continue
        counts[('raw_good' if a else 'raw_weak')+('_final_good' if f else '_final_weak')]+=1
        counts[('plan_good' if p else 'plan_weak')+('_final_good' if f else '_final_weak')]+=1
        req=ok(e,vs['requested'])
        counts[('request_good' if req else 'request_weak')+('_final_good' if f else '_final_weak')]+=1
        fixed=ok(e,vs['bound'])
        if fixed is not None:counts[('plan_good' if p else 'plan_weak')+('_bound_good' if fixed else '_bound_weak')]+=1
    return dict(counts=counts,source_of_plan_loss=sources)
def describe(rows,n):
    stats=distances(rows,n);replayed=[r['rolling'][n] for r in rows if 'rolling' in r]
    completed=[r for r in rows if values(r,n)['final'] is not None]
    return dict(windows=len(rows),episodes=len(set(r['ep'] for r in rows)),stages=stats,transitions=transitions(rows,n),
        sources=dict(Counter(r[n]['source'] for r in rows)),rolling_clips=med(a['clips'] for a in replayed),
        interrupted=dict(Counter('plan_reconstruction' if a['guard'].startswith('plan:') else 'target_guard' for a in replayed if a['guard'])),
        rolling_plans=med(a['plans'] for a in replayed),rolling_with_close=sum(bool(a['calls']) for a in replayed),
        raw_command_xy_error=med(diffxy(r['raw'],r['command']) for r in rows),
        raw_command_cos60=sum(angle(sub(r['command'],r['anchor']),sub(r['raw'],r['anchor'])) is not None and
            angle(sub(r['command'],r['anchor']),sub(r['raw'],r['anchor']))<=60 for r in rows),
        raw_command_direction_defined=sum(angle(sub(r['command'],r['anchor']),sub(r['raw'],r['anchor'])) is not None for r in rows),
        expert_command_xy_error=med(diffxy(r['expert'],r['command']) for r in rows),
        final_vs_last_observation_along=med(projection(r['expert'],sub(values(r,n)['final'],r['feedback7'])) for r in completed),
        last_observation_along=med(projection(r['expert'],r['feedback7']) for r in completed),
        expert_remaining_along=med(projection(r['expert'],sub(r['expert'],r['feedback7'])) for r in completed),
        anchor_xy=med(size(r['anchor']) for r in rows),
        feedback7_xy_error=med(diffxy(r['expert'],r['feedback7']) for r in rows),dt_ms=med(r['dt_ms'] for r in rows))
