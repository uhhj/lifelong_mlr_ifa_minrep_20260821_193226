"""Local Q/S paired evidence review; no model, server or environment execution."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,statistics,datetime

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
ROOT=WORK/'execution/20261031_full_q_vs_latest_s_two_suites'
FINAL=ROOT/'final_readonly_20261101_032229'
BINDINGS={}

def read(p,parse=True):
    raw=p.read_bytes()
    BINDINGS[p.relative_to(WORK).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw) if parse else raw.decode('utf-8-sig')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def light(e,n):
    x=e['light'][n]
    assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
    return json.loads(x['text'])

def metrics(m):
    return dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
                FWT=sum(m[t,t] for t in range(6,10))/4,
                NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
                OldFinal=sum(m[9,t] for t in range(6,9))/3,
                FinalAll=sum(m[9,t] for t in range(6,10))/4)

def main():
    acc=read(ROOT/'ACCEPTANCE.json'); audit=read(FINAL/'LOCAL_FINAL_AUDIT.json')
    comp=read(FINAL/'PRIMARY_S_COMPARISON.json'); qe=read(FINAL/'EVIDENCE.json')
    assert acc['audit_sha256']==comp['source_Q_final_sha256']==sha(FINAL/'LOCAL_FINAL_AUDIT.json')
    assert acc['comparison_sha256']==sha(FINAL/'PRIMARY_S_COMPARISON.json')
    assert audit['evidence_sha256']==sha(FINAL/'EVIDENCE.json')
    assert acc['report_sha256']==sha(ROOT/'FINAL_REPORT.md')
    assert audit['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
    assert audit['formal_episodes']==1200 and audit['physical_training_optimizer_calls']==544800
    for n in ['FINAL_REPORT.md','PROTOCOL.md']:
        read(ROOT/n,False)
    read(ROOT/'leakage_audit_20261031/REPORT.md',False)
    old=read(WORK/'execution/20261031_publication_readiness/EVIDENCE_REVIEW.json')
    read(WORK/'execution/20261031_publication_readiness/REPORT.md',False)
    read(WORK/'execution/20261026_shbr_novelty_review/REPORT.md',False)
    folders={'GOAL':'20261029_support_gate_full_s_goal/final_readonly_20261030_111546','OBJECT':'20261028_support_gate_full_s_object/final_readonly_20261029_201235'}
    cases={}; negative=[]; total=0
    for suite,folder in folders.items():
        p=WORK/'execution'/folder
        se=read(p/'EVIDENCE.json'); sa=read(p/'LOCAL_FINAL_AUDIT.json')
        assert comp['source_S'][suite]['audit_sha256']==sha(p/'LOCAL_FINAL_AUDIT.json')
        assert comp['source_S'][suite]['evidence_sha256']==sa['evidence_sha256']==sha(p/'EVIDENCE.json')
        for seed in [41,42,43]:
            case=f'{suite}_{seed}'; data=comp['cases'][case]
            key=lambda x:(x['stage'],x['task'],x['initial_index'])
            qr={key(x):x for x in light(qe,case+'/OUTCOMES.json')}
            sr={key(x):x for x in light(se,f'seed{seed}/OUTCOMES.json')}
            pairs={key(x):x for x in data['paired_outcomes']}
            assert qr.keys()==sr.keys()==pairs.keys() and len(pairs)==200
            for k,x in pairs.items():
                assert x['Q']==qr[k]['success'] and x['S']==sr[k]['success']
                for f in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt']:
                    assert qr[k][f]==sr[k][f]
                for arm,record in [('Q',qr[k]),('S',sr[k])]:
                    assert x[arm+'_cp_sha256']==record['cp_sha256']
            matrices={'Q':{},'S':{}}; sf=fs=0
            for s in range(6,10):
                for t in range(6,s+1):
                    rows=[v for k,v in pairs.items() if k[:2]==(s,t)]
                    assert len(rows)==20 and {x['initial_index'] for x in rows}==set(range(20))
                    for arm in matrices: matrices[arm][s,t]=F(sum(x[arm] for x in rows),20)
                    delta=100*(matrices['S'][s,t]-matrices['Q'][s,t])
                    if delta<0: negative.append(dict(case=case,cell=f'{s}_{t}',delta_pp=float(delta)))
                    sf+=sum(x['Q'] and not x['S'] for x in rows)
                    fs+=sum(x['S'] and not x['Q'] for x in rows)
            result={arm:{k:float(v*100) for k,v in metrics(m).items()} for arm,m in matrices.items()}
            for arm in result:
                for k,v in result[arm].items(): assert abs(v-data[arm+'_metrics'][k])<1e-9
            cases[case]=dict(**result,Q_success_S_fail=sf,Q_fail_S_success=fs)
            total+=len(pairs)
    suites={}
    for suite in folders:
        suites[suite]={}
        for metric in ['AUC','FWT','NBT','OldFinal','FinalAll']:
            a={arm:statistics.mean(cases[f'{suite}_{s}'][arm][metric] for s in [41,42,43]) for arm in ['Q','S']}
            a['S_minus_Q_pp']=a['S']-a['Q']
            assert abs(a['S_minus_Q_pp']-comp['suites'][suite][metric]['S_minus_Q_pp'])<1e-9
            suites[suite][metric]=a
    assert total==1200 and len(negative)==13
    out=dict(status='PASS_POST_Q_LOCAL_EVIDENCE_REVIEW',observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             paired_episodes=total,suites=suites,cases=cases,negative_cells=negative,source_bindings=BINDINGS,
             scope='Existing acceptance SHA binding and outcome recomputation; no repeated CPU checkpoint load',
             new_optimizer=0,new_environment=0,remote_experiment_commands=0)
    (HERE/'EVIDENCE_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],suites=suites,negative_cells=len(negative)),ensure_ascii=False))

if __name__=='__main__': main()
