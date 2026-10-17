import json
from fractions import Fraction as F
from pathlib import Path
EXPECTED={(6,6):6,(7,6):14,(7,7):17,(8,6):4,(8,7):19,(8,8):20,(9,6):6,(9,7):18,(9,8):20,(9,9):9}
def calculate(counts):
    r={k:F(v,20) for k,v in counts.items()}
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/ (10-t) for t in range(6,10))/4,
      FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
      OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def read(root):
    counts={};records={}
    for s in range(6,10):
        for t in range(6,s+1):
            rows=[json.loads(l) for l in (Path(root)/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
            assert len(rows)==20 and {r['initial_state_index'] for r in rows}==set(range(20))
            for r in rows:
                assert r['stage']==s and r['task']==t and r['epoch']==99 and r['seed']==42 and type(r['success']) is bool
                records[s,t,r['initial_state_index']]=r
            counts[s,t]=sum(r['success'] for r in rows)
    return counts,records
def serial(metrics):return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics.items()}
def comparison(base,new):
    a,ar=read(base);b,br=read(new);assert a==EXPECTED
    for k in ar:
        for name in ('initial_state_sha256','modal_sha256'):assert ar[k][name]==br[k][name],(k,name)
    am=calculate(a);bm=calculate(b)
    gate=dict(AUC=bm['AUC']>am['AUC'],FWT=bm['FWT']>=am['FWT'],NBT=bm['NBT']<am['NBT'],OldFinal=bm['OldFinal']>=am['OldFinal'])
    return dict(status='METRICS_RECOMPUTED_NOT_FULL_ACCEPTANCE',baseline=serial(am),new=serial(bm),delta_pp={k:float((bm[k]-am[k])*100) for k in am},baseline_counts={f'{s},{t}':v for (s,t),v in a.items()},new_counts={f'{s},{t}':v for (s,t),v in b.items()},progression_gate=gate,all_gates_pass=all(gate.values()),auto_extend=False)
