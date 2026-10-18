"""Read-only validation of the finished Random44 arm. No experiment execution."""
import json, hashlib
from fractions import Fraction as F
from pathlib import Path
R=Path(__file__).resolve().parent
for p,v in json.loads((R/'MANIFEST.json').read_text()).items():
    assert hashlib.sha256((R/p).read_bytes()).hexdigest()==v['sha256']
e=R/'seed44/random/evaluation';r={}
for s in range(6,10):
    for t in range(6,s+1):
        records=[json.loads(x) for x in (e/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
        assert len(records)==20 and {x['initial_state_index'] for x in records}==set(range(20))
        assert all((x['seed'],x['stage'],x['task'],x['epoch'])==(44,s,t,99) and type(x['success']) is bool for x in records)
        events=[json.loads(x) for x in (e/f'attempts_stage{s}_task{t}.jsonl').read_text().splitlines()]
        assert not any(x['event']=='technical_exception' for x in events)
        starts=[x for x in events if x['event']=='vector_attempt_start']
        ends=[x for x in events if x['event']=='vector_attempt_complete']
        assert len(starts)==len(ends)==2
        for g in (0,1):
            a=next(x for x in starts if x['group']==g);b=next(x for x in ends if x['group']==g)
            assert a['env_seed']==b['env_seed']==44+g and a['initial_indices']==list(range(g*10,g*10+10))
            assert b['steps']<=600
            for i,v in enumerate(b['successes']):assert v==next(x['success'] for x in records if x['initial_state_index']==g*10+i)
        r[s,t]=F(sum(x['success'] for x in records),20)
m={'AUC':sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,'FWT':sum(r[t,t] for t in range(6,10))/4,'NBT':sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3}
print(json.dumps({'status':'PARTIAL_EXPERIMENT','completed_arm':'seed44_random','episodes':200,'vector_attempts':20,'technical_exceptions':0,'metrics':{k:{'fraction':str(v),'percent':float(100*v)} for k,v in m.items()},'matrix':[[float(100*r[s,t]) if t<=s else None for t in range(6,10)] for s in range(6,10)]},indent=2))
