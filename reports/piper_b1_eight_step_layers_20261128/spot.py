"""Branch the archived expert approach at each labelled start; advance the real predictor, guard and placement for eight ticks, without devices."""
from common import *
from bus import bounded_target
from b1_placement import B1Placement
import ast
import copy
from collections import deque
from functools import lru_cache

def detector_type():
    path=work/'piper_test_console_pika_b1_full_v1/runner.py'
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='GraspIntent')
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['GraspIntent']

def fork(m):
    b=copy.copy(m)
    for k in ('history','raw','heat','features'):
        setattr(b,k,getattr(m,k).copy())
    for k in ('cue','relative','tracker','wrist_position','trace','details','visual','anchor'):
        if hasattr(m,k):setattr(b,k,copy.deepcopy(getattr(m,k)))
    b.chunk=None
    return b

def run(group,micro=False):
    intent_cls=detector_type();models_=models();m=models_['candidate']
    window_file=dest/('windows_'+group+'.json')
    reference=read(window_file)
    name=('micro_' if micro else 'rolling_')+group
    part=dest/(name+'_partial.json')
    signature=rp.sha(Path(__file__))+rp.sha(window_file)
    prior=read(part) if part.exists() else {}
    out=prior.get('rows',{}) if prior.get('signature')==signature else {}
    for ep,rec in subjects(group):
        if ep in out:continue
        c=rec['close'];good,stamps,fb,cmd,x,cmdx,ct,goal=recorded(rec,m)
        for v in models_.values():rp.reset(v)
        rows={z['t']:z for z in reference[ep]['windows']}
        pick={t for t,r in rows.items() if r['phase']!='other' or t==c-18}
        if micro:
            pick=set()
            for t,r in rows.items():
                h=np.linalg.norm(x[t+8,:2]-x[t,:2])
                if not (r['grid'] and 1<=h<3):continue
                ok=True
                for center in (goal,rec['visual_goal']):
                    delta=center-x[t,:2];d0=np.linalg.norm(delta)
                    d8=np.linalg.norm(center-x[t+8,:2])
                    ok=ok and 12<d0<=60 and d0-d8>=1 and delta@(x[t+8,:2]-x[t,:2])>=.8*d0*h
                if ok:pick.add(t)
            if not pick:continue
        place={n:B1Placement() for n in models_};ints={n:intent_cls() for n in models_}
        @lru_cache(maxsize=12)
        def images(i):return [rp.imaging(rec,i,k) for k in ('overhead','wrist')]
        out[ep]={}
        for t in range(max(pick)+1):
            if not good[t]:
                for n,v in models_.items():
                    rp.reset(v);place[n]=B1Placement();ints[n]=intent_cls()
                continue
            anchor=cmd[t-1] if t and good[t-1] else fb[t]
            if t in pick:
                res={};out[ep][t]=res
                for n,v in models_.items():
                    b=fork(v);g=copy.deepcopy(place[n]);intent=copy.deepcopy(ints[n])
                    last=anchor.tolist();clips=plans=0;calls=[];err=None;diff=0
                    for k in range(8):
                        i=t+k;s=int(stamps[i])
                        try:pred=visual_step(b,images(i),s,fb[i].tolist(),last,g.phase)
                        except ValueError as ex:
                            if k==0:raise
                            err='plan: '+str(ex)[:105];break
                        if k==0:
                            diff=max(np.max(abs(xyz(b,b.chunk[-1])-x[t]-rows[t][n]['plan'])),
                                     np.max(abs(xyz(b,b.base_plan[-1])-x[t]-rows[t]['raw'])))
                            assert diff<.003,(ep,t,n,diff)
                        plans+=int(b.trace['chunk_stamp']==s)
                        intent.observe(s,fb[i,6],s,b.intent_widths)
                        try:target=bounded_target(pred,fb[i].tolist(),last,(s-int(stamps[i-1]))/1e9,b3_tracking=True,arm_only=True)
                        except ValueError as ex:
                            err='guard: '+str(ex)[:103];break
                        clips+=int(np.max(abs(np.array(target)-np.array(pred[:6])))>1e-7)
                        width=g.next_goal(s,intent.fired,x[i],fb[i,6],vision=b.visual)
                        if width is not None:calls.append([k,width])
                        last=list(target)+[last[6] if width is None else width]
                    res[n]=dict(final=None if err else vec(xyz(b,last)-x[t]),requested=None if err else vec(xyz(b,pred)-x[t]),
                                plans=plans,clips=clips,guard=err,calls=calls,initial_check_mm=round(float(diff),6))
            for n,v in models_.items():
                s=int(stamps[t]);visual_step(v,images(t),s,fb[t].tolist(),anchor.tolist())
                if v.trace['ready']:
                    ints[n].observe(s,fb[t,6],s,v.intent_widths)
                    place[n].next_goal(s,False,x[t],fb[t,6],vision=v.visual)
        dump(part.name,dict(signature=signature,rows=out))
        if len(out)%4==0:print(group,len(out),'episodes',flush=True)
    dump(name+'.json',out);print(group,'rolling complete',len(out),sum(map(len,out.values())),flush=True)
if __name__=='__main__':
    if sys.argv[6]=='identity':
        import identity_check
    else:
        import fcntl
        cohort=sys.argv[6].removeprefix('recover_')
        assert cohort in ('archive','regression','micro_archive','micro_regression')
        with (dest/(cohort+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            group=cohort.removeprefix('micro_');micro=cohort.startswith('micro_')
            final=dest/(('micro_' if micro else 'rolling_')+group+'.json')
            if final.exists():
                full=read(final);episodes=read(dest/('windows_'+group+'.json'))
                assert (full.keys()<=episodes.keys()) if micro else (full.keys()==episodes.keys())
            else:run(group,micro)
