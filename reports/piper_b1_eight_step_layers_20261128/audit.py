"""All eight-command pre-grasp windows; independent expert-only phase labels."""
from common import *
from bus import bounded_target
def projected(m,plan,fb,cmd,stamps,t):
    last=cmd[t-1].tolist();clipped=0
    for k,request in enumerate(plan):
        i=t+k
        try:q=bounded_target(request.tolist(),fb[i].tolist(),last,(int(stamps[i])-int(stamps[i-1]))/1e9,b3_tracking=True,arm_only=True)
        except ValueError as ex:return None,k,str(ex)[:110]
        clipped+=int(np.max(abs(np.array(q)-request[:6]))>1e-7)
        last=list(q)+[last[6]]
    return xyz(m,last),clipped,None

def run(group):
    models_=models();m=models_['candidate'];result={};excluded=Counter()
    reference=read(frozen/('continuous_'+group+'.json'))
    for ep,rec in subjects(group):
        c=rec['close'];good,stamps,fb,cmd,x,cmdx,ct,goal=recorded(rec,m)
        for v in models_.values():rp.reset(v)
        entries=[];bad=Counter();check=None
        for t in range(c-7):
            if not good[t]:
                for v in models_.values():rp.reset(v)
                if t>=7:bad['invalid']+=1
                continue
            views=[rp.imaging(rec,t,k) for k in ('overhead','wrist')]
            anchor=cmd[t-1] if t and good[t-1] else fb[t]
            for v in models_.values():visual_step(v,views,stamps[t],fb[t].tolist(),anchor.tolist())
            if t<7:continue
            if not good[t-7:t+9].all():bad['invalid']+=1;continue
            delta=np.diff(stamps[t-7:t+9])/1e6
            if min(delta)<=0 or max(delta)>70:bad['discontinuity']+=1;continue
            if not all(v.trace['ready'] for v in models_.values()):bad['not_ready']+=1;continue
            if min(cmd[t:t+8,6])<=55:bad['expert_closing']+=1;continue
            start=x[t];expert=x[t+8]-start;xy=start[:2]
            width=float(min(fb[t,6],fb[t+8,6]))
            label=classify(start,x[t+8],goal,width)
            row=dict(t=t,grid=(c-8-t)%8==0,phase=label,image_phase=classify(start,x[t+8],rec['visual_goal'],width),
                feedback7_phase=classify(start,x[t+7],goal,width),phase_lo=classify(start,x[t+8],goal,width,.8),phase_hi=classify(start,x[t+8],goal,width,1.2),
                expert=vec(expert),feedback7=vec(x[t+7]-start),command=vec(cmdx[t+7]-start),anchor=vec(cmdx[t-1]-start),
                d0=round(float(np.linalg.norm(goal-xy)),4),d8=round(float(np.linalg.norm(goal-x[t+8,:2])),4),dt_ms=round((int(stamps[t+8])-int(stamps[t]))/1e6,1))
            for name,v in models_.items():
                base,plan,status=fresh(v,anchor)
                raw=xyz(v,base[-1])-start;p=xyz(v,plan[-1])-start
                if 'raw' not in row:row['raw']=vec(raw)
                else:assert np.max(abs(np.array(row['raw'])-raw))<.001
                assert abs(raw[2]-p[2])<.001 and np.array_equal(base[:,6],plan[:,6])
                bounded,k,err=projected(v,plan,fb,cmd,stamps,t)
                wp=getattr(v,'relative',None)
                wg=None if wp is None else getattr(wp,'wrist_goal',None)
                source='raw_pass' if not status['applied'] else 'planar_correction'
                if status['phase']=='hold_unobserved_xy':source='no_target_hold'
                elif status['applied'] and wg is not None:
                    source='wrist_deadband' if np.linalg.norm(wg-xy)<=wp.uncertainty else 'wrist_guidance'
                row[name]=dict(plan=vec(p),bound=None if err else vec(bounded-start),clips=k,blocked=err is not None,source=source)
                if t==c-18 and name=='candidate':
                    old_goal=rec['visual_goal'] if group=='regression' else goal
                    value=np.linalg.norm(start[:2]-old_goal)-np.linalg.norm(start[:2]+p[:2]-old_goal)
                    check=float(abs(value-reference[ep]['natural'][1]))
                    assert check<.003,(ep,check)
            entries.append(row)
        assert check is not None and entries,(ep,check)
        result[ep]=dict(close=c,contact=ct,reference_gap_mm=round(float(np.linalg.norm(goal-rec['visual_goal'])),3),anchor_check_mm=check,
            input_sha=rp.sha(rec['folder']/'alignment.jsonl'),excluded=dict(bad),windows=entries)
        if len(result)%8==0:print(group,len(result),'episodes',flush=True)
    dump('windows_'+group+'.json',result)
    print(group,'done',len(result),sum(len(r['windows']) for r in result.values()),flush=True)
if __name__=='__main__':run(sys.argv[6])
