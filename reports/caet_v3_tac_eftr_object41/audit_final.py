"""Local exact closed-loop recomputation and independent mechanism ledger audit."""
import json,sys,hashlib,math,random
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from formal_audit import audit_formal
seed=41;variant=sys.argv[2];assert variant in ('TAC','EFTR','ALIGN')
p=Path(sys.argv[1]);manifest=json.loads((p/'PACKAGE_MANIFEST.json').read_text())
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,r in manifest['entries'].items():assert sha(p/name)==r['sha256'] and (p/name).stat().st_size==r['bytes']
assert not manifest['residual_workers']
C=p/'current';B=p/'baseline';V=p/'v1'
assert (p/'launcher/supervisor.exit').read_text().strip()=='0' and (C/'pipeline.exit').read_text().strip()=='0'
assert js(p/'launcher/ENTRY.json')['seed']==41 and js(p/'launcher/ENTRY.json')['variant']==variant
method={'TAC':'CAET-v3-TAC','EFTR':'RA-EFTR-v3','ALIGN':'RA-TAC-Align'}[variant]
a=js(C/'AUDIT.json');process=js(C/'PROCESS_EXIT_ACCEPTANCE.json');gate=js(C/'GPU_PREFLIGHT.json')
assert a['status']==process['status']==gate['status']=='PASS' and a['method']==method and a['all_inputs_unchanged']
assert process['audit_sha256']==sha(C/'AUDIT.json') and not process['residual_workers']
assert gate['environment_steps']==0 and gate['proof']['disabled_full_state_exact_RA'] and gate['proof']['active_serialized_full_state_restore_exact']
assert js(C/'ra_coreset/INITIAL_STATE.json')==js(B/'ra_coreset/INITIAL_STATE.json')
if variant=='EFTR':assert js(C/'TASK6_TENSOR_PARITY.json')['equal']
h=js(C/'ra_coreset/epochs.json');assert len(h)==400;steps=0;counts=Counter();headsteps=0
for t in range(6,10):
 md=js(C/'ra_coreset/dataset_manifests'/('task%d.json'%t));rows=md['rows'];n=len(rows);assert n==[1562,2261,3081,3620][t-6]
 assert [r['index'] for r in rows]==list(range(n)) and len({r['path'] for r in rows})==n
 current=js(C/'ra_coreset/storage'/('task%d.json'%t));base=js(B/'ra_coreset/storage'/('task%d.json'%t))
 assert current['K']==base['K']=={6:781,7:740,8:780,9:660}[t]
 assert current['selection']['selected_ids']==base['selection']['selected_ids']
 assert [(Path(x['source']).name,x['destination'],x['sha256']) for x in current['selected']]==[(Path(x['source']).name,x['destination'],x['sha256']) for x in base['selected']]
 assert js(C/'ra_coreset'/('ifa_task%d.json'%t))==js(B/'ra_coreset'/('ifa_task%d.json'%t))
 assert next(x for x in a['checkpoints'] if x['task']==t)['strict_loaded_keys']==100
 positive=md['positive'];negative=md['negative'];oldtasks={j:sorted([r['index'] for r in rows if r['task']==j],key=lambda i:rows[i]['sample']) for j in range(6,t)}
 old=[v[k] for k in range(max([len(v) for v in oldtasks.values()] or [0])) for j,v in sorted(oldtasks.items()) if k<len(v)];cursor=0
 for ep in range(100):
  row=h[(t-6)*100+ep];assert (row['task'],row['epoch'],row['batches'])==(t,ep,math.ceil(n/10));steps+=row['batches'];assert row['steps']==steps
  stats=js(C/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,ep)));assert [x['step'] for x in stats]==list(range(steps-row['batches'],steps))
  assert sorted(i for s in stats for i in s['batch_ids'])==list(range(n))
  for s in stats:
   ids=s['batch_ids'];v=sum(rows[i]['bc_valid'] for i in ids) if variant!='EFTR' else len(ids)
   assert s['bc_valid']==v and s['bc_invalid']==len(ids)-v and s['committed_updates']==1
   counts['main_updates']+=1;counts['bc_exposures']+=v;counts['invalid_bc_exposures']+=len(ids)-v;counts['optimizer_trials']+=s['optimizer_trials']
   if variant=='TAC':
    em=sum(rows[i]['effect_valid'] for i in ids);anchors=[i for i in ids if positive.get(str(i)) and negative.get(str(i))];partners=[]
    for i in anchors:
     ps=positive[str(i)];ns=negative[str(i)];partners.append([ps[s['step']%len(ps)]]+[ns[(s['step']+k)%len(ns)] for k in range(min(4,len(ns)))])
    assert s['effect_samples']==em and s['anchor_ids']==anchors and s['partner_ids']==partners and s['extra_partner_forwards']==sum(len(x) for x in partners)
    headsteps+=int(bool(em or anchors));assert s['head_steps']==headsteps and s['head_updates']==int(bool(em or anchors))
    counts['head_updates']+=s['head_updates'];counts['effect_exposures']+=em;counts['cross_anchors']+=len(anchors);counts['partner_forwards']+=s['extra_partner_forwards']
   if variant=='EFTR' and t>6:
    ix=[old[(cursor+k)%len(old)] for k in range(min(32,len(old)))];cursor=(cursor+len(ix))%len(old);assert ix==s['old_check_ids'] and cursor==s['cursor']
    scores=s['scores'];c0=scores[0]['C'];valid=[i for i,z in enumerate(scores) if c0 is None or z['C']<=c0+.02];assert 0 in valid
    budget=[i for i in valid if scores[i]['R']<=.10];chosen=min(budget or valid,key=lambda i:(scores[i]['R'],scores[i]['C'] or 0.,i))
    assert chosen==s['selected'] and s['budget_unmet']==(not bool(budget)) and s['optimizer_trials']==3
    counts['budget_unmet']+=s['budget_unmet'];counts['selected_'+str(chosen)]+=1;counts['old_check_exposures']+=len(ix)
assert steps==105500 and dict(counts)==a['auxiliary']['counts']
def load(root,pattern):return [json.loads(line) for f in sorted((root/'ra_coreset/evaluation').glob(pattern)) for line in f.read_text().splitlines()]
trials=load(C,'trials*.jsonl');events=load(C,'attempts*.jsonl');cells={f.name:js(f) for f in (C/'ra_coreset/evaluation').glob('cell*.json')}
formal=audit_formal(dict(trials=trials,attempts=events,cells=cells),seed)
assert formal['records']==200 and formal['started_vectors']==formal['completed_vectors']==20 and formal['completed_cells']==10
def matrix(rows):
 assert len(rows)==200;by={(r['stage'],r['task'],r['trial']):r for r in rows};assert len(by)==200
 rates={}
 for s in range(6,10):
  for t in range(6,s+1):
   values=[by[s,t,i] for i in range(20)];assert all(r['seed']==seed and type(r['success']) is bool for r in values)
   rates[s,t]=F(sum(r['success'] for r in values),20)
 return rates,by
def metrics(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
rates,new=matrix(trials);base,old=matrix(load(B,'trials*.jsonl'));v1,_=matrix(load(V,'trials*.jsonl'))
for key,r in new.items():
 assert r['initial_state_sha256']==old[key]['initial_state_sha256'] and r['modal_sha256']==old[key]['modal_sha256']
 assert r['checkpoint_sha256']==next(c['sha256'] for c in a['checkpoints'] if c['task']==r['stage'])
m,b,v=metrics(rates),metrics(base),metrics(v1)
assert all(F(a['metrics'][k]['fraction'])==x and F(a['baseline'][k]['fraction'])==b[k] for k,x in m.items())
out=dict(status='COMPLETED_LOCAL_INDEPENDENT_PASS',method=method,suite='OBJECT',seed=41,metrics=pack(m),ra=pack(b),caet_v2_baseline=pack(v),delta_ra=pack({k:m[k]-b[k] for k in m}),delta_v2_baseline=pack({k:m[k]-v[k] for k in m}),matrix=[dict(stage=s,task=t,current=str(rates[s,t]),ra=str(base[s,t]),v2_baseline=str(v1[s,t])) for s,t in rates],counts=dict(counts),formal=formal,main_steps=steps,preflight=gate,main_training_seconds=sum(r['seconds'] for r in h),package_manifest_sha256=sha(p/'PACKAGE_MANIFEST.json'))
(p/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
report='# '+method+' OBJECT seed41 final acceptance\n\nIndependent exact full episode matrix and mechanism ledger PASS. Single-seed exploratory test.\n\n|Metric|RA %|Original v2 %|New %|New-RA pp|\n|---|---:|---:|---:|---:|\n'
for k in m:report+=f'|{k}|{float(100*b[k]):.6f}|{float(100*v[k]):.6f}|{float(100*m[k]):.6f}|{float(100*(m[k]-b[k])):+.6f}|\n'
report+='\nFWT is trained-diagonal performance; NBT lower is better. TAC/ALIGN change BC targets and valid exposure. EFTR retains original BC. No closed-loop guarantee or automatic seed expansion. All extra computation counted.\n'
(p/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(out,indent=2))
