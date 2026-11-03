"""Read-only descriptive comparison; explicitly not final CPU acceptance."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,hashlib,statistics
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['snapshot_sha256']==sha(D/'EVIDENCE.json')
b=js(R/'S_ACCEPTANCE.json')['sources']['GOAL'];B=Path(b['final_path'])
assert sha(B/'EVIDENCE.json')==b['evidence_sha256'] and sha(B/'LOCAL_FINAL_AUDIT.json')==b['audit_sha256']
se=js(B/'EVIDENCE.json');sa=js(B/'LOCAL_FINAL_AUDIT.json')
assert sa['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and sa['formal_episodes']==600
def get(ev,key):
    v=ev['light'][key];assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'];return json.loads(v['text'])
def metrics(rows):
    r={}
    for s in range(6,10):
        for t in range(6,s+1):
            v=[x for x in rows if (x['stage'],x['task'])==(s,t)];assert len(v)==20
            r[s,t]=F(sum(x['success'] for x in v),20)
    m=dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4,NewFinal=r[9,9])
    return m,r
cases={};pairs=[];cells=[]
for seed in (41,42,43):
    c='L005_'+str(seed);p=c+'/';sp='seed%d/'%seed
    assert a['counts'][c]['formal_episodes']==200 and a['counts'][c]['epochs']==400
    for x in ['train_S.exit','evaluate.exit']:assert e['light'][p+x]['text'].strip()=='0'
    cr=get(e,p+'OUTCOMES.json');sr=get(se,sp+'OUTCOMES.json');key=lambda x:(x['stage'],x['task'],x['initial_index'])
    cc={key(x):x for x in cr};ss={key(x):x for x in sr};assert len(cc)==len(cr)==len(ss)==len(sr)==200 and cc.keys()==ss.keys()
    ci=get(e,p+'INITIAL_S.json');si=get(se,sp+'INITIAL_S.json')
    for k in ['tensors','rng','official_cp_sha256','parent_initial_sha256']:assert ci[k]==si[k]
    for k in sorted(cc):
        for f in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt']:assert cc[k][f]==ss[k][f],(c,k,f)
        pairs.append(dict(seed=seed,stage=k[0],task=k[1],initial_index=k[2],L005=cc[k]['success'],L010=ss[k]['success']))
    cm,ct=metrics(cr);sm,st=metrics(sr)
    for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:assert str(sm[k])==sa['seeds'][str(seed)]['metrics'][k]['fraction']
    cases[c]=dict(L005={k:float(v*100) for k,v in cm.items()},L010={k:float(v*100) for k,v in sm.items()},delta_L005_minus_L010_pp={k:float((cm[k]-sm[k])*100) for k in cm},hardware=dict(L005=ci['hardware'],L010=si['hardware']))
    for s,t in ct:cells.append(dict(seed=seed,stage=s,task=t,L005=float(ct[s,t]*100),L010=float(st[s,t]*100),delta_pp=float((ct[s,t]-st[s,t])*100)))
means={k:{label:statistics.mean(v[label][k] for v in cases.values()) for label in ['L005','L010','delta_L005_minus_L010_pp']} for k in cm}
out=dict(status='DESCRIPTIVE_600_PAIRED_OUTCOMES_PENDING_FINAL_CPU_ACCEPTANCE',observation=a['observation'],evidence_sha256=a['snapshot_sha256'],baseline=b,cases=cases,means=means,cells=cells,paired_outcomes=pairs,L005_success_L010_fail=sum(x['L005'] and not x['L010'] for x in pairs),L005_fail_L010_success=sum(not x['L005'] and x['L010'] for x in pairs),cell_wins=sum(x['delta_pp']>0 for x in cells),cell_ties=sum(x['delta_pp']==0 for x in cells),cell_losses=sum(x['delta_pp']<0 for x in cells),limits=['Descriptive recomputation only; full CPU checkpoint/cache/state acceptance not yet run.','Hardware migration and development-set reuse/pretraining provenance limitations remain.','No lambda selection or additional training/evaluation.'])
(D/'L005_OBSERVED_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k not in ['paired_outcomes','cells']},ensure_ascii=False,indent=2))
