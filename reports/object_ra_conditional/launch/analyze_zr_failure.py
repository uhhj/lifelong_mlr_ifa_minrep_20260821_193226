"""Read-only failure decomposition from already completed logs. No ML imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,datetime,statistics
R=Path(__file__).resolve().parent;D=R/'completion_20261020/download';OUT=R/'analysis_20261020_zrv2'
OUT.mkdir(exist_ok=True);inputs={}
def read(p):
 raw=p.read_bytes();inputs[str(p.relative_to(R))]=hashlib.sha256(raw).hexdigest();return raw.decode('utf-8')
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(x) for x in read(p).splitlines()]
def pack(x):return dict(fraction=str(x),pp=float(x*100))
accepted=js(R/'completion_20261020/LOCAL_FINAL_ACCEPTANCE.json');assert accepted['status']=='PASS'
for p in [R/'server_auto/zr_template/source/zr2_exp'/n for n in ['calibrate.py','model.py','diagnostics.py','evaluate.py']]+[R/'server_auto/zr_template/attachment/reference_core.py',R/'source/coreset_exp/eval_metric.py']:
 read(p)
cells=[];pairs=[];epochs=[];structs=[];rng_groups=[];costs=[];rates={};configs=[]
for seed in [41,42,43]:
 root=D/'evidence'/('OBJECT_RA_seed%d'%seed);rate={}
 ca=js(root/'ra_coreset/evaluation/resolved_actual.json');cb=js(root/'evaluation/resolved_actual.json')
 dirs=[ca.pop('experiment_dir'),cb.pop('experiment_dir')]
 assert ca==cb,('Unexpected evaluation configuration difference',seed)
 configs.append(dict(seed=seed,only_difference='experiment_dir',values=dirs,scientific_config_equal=True))
 erows=js(root/'epochs.json');sr=jl(root/'STRUCTURE.jsonl');structs.extend(dict(seed=seed,**x) for x in sr)
 assert len(sr)==16 and all(x['passed'] for x in sr)
 cost=js(root/'COST_ISOLATION.json');costs.append(dict(seed=seed,**cost))
 for stage in range(6,10):
  ep=[r for r in erows if r['task']==stage];assert len(ep)==10
  st=next(r for r in sr if r['stage']==stage and r['label']=='stage_start')
  en=next(r for r in sr if r['stage']==stage and r['label']=='stage_end')
  epochs.append(dict(seed=seed,stage=stage,epoch0_nll=ep[0]['loss'],epoch9_nll=ep[-1]['loss'],nll_delta=ep[-1]['loss']-ep[0]['loss'],
                     delta_start=st['delta_max'],delta_end=en['delta_max'],saturation=en['saturation_fraction'],samples=en['samples']))
  # Track action-draw budget before each vector group within each reset stage.
  offset_a=0;offset_b=0
  for task in range(6,stage+1):
   name='trials_stage%d_task%d.jsonl'%(stage,task)
   pa=jl(root/'ra_coreset/evaluation'/name);pb=jl(root/'evaluation'/name)
   assert len(pa)==len(pb)==20
   a={x['trial']:x for x in pa};b={x['trial']:x for x in pb};assert set(a)==set(b)==set(range(20))
   ct=Counter()
   for i in range(20):
    assert a[i]['initial_state_sha256']==b[i]['initial_state_sha256'] and a[i]['checkpoint_sha256']==b[i]['parent_checkpoint_sha256']
    tag=('S' if a[i]['success'] else 'F')+('S' if b[i]['success'] else 'F');ct[tag]+=1
    pairs.append(dict(seed=seed,stage=stage,task=task,initial=i,initial_hash=a[i]['initial_state_sha256'],parent=a[i]['success'],new=b[i]['success'],pair=tag))
   for tag in ['SS','SF','FS','FF']:ct[tag]+=0
   old=F(sum(x['success'] for x in pa),20);new=F(sum(x['success'] for x in pb),20);delta=new-old
   rate[stage,task]=(old,new)
   cells.append(dict(seed=seed,stage=stage,task=task,parent=float(old*100),new=float(new*100),delta_pp=float(delta*100),
                     auc_contribution_pp=float(delta/F(4*(10-task))*100),**ct))
   for g in [0,1]:
    aa={a[i]['vector_group_steps'] for i in range(10*g,10*g+10)};bb={b[i]['vector_group_steps'] for i in range(10*g,10*g+10)}
    assert len(aa)==len(bb)==1
    sa=next(iter(aa));sb=next(iter(bb))
    rng_groups.append(dict(seed=seed,stage=stage,task=task,group=g,parent_steps=sa,new_steps=sb,
                           parent_prior_action_calls=offset_a,new_prior_action_calls=offset_b,draw_budget_aligned=offset_a==offset_b))
    offset_a+=sa;offset_b+=sb
  for name in ['ra_coreset/evaluation/attempts_stage%d_task%d.jsonl'%(stage,t) for t in range(6,stage+1)]+['evaluation/attempts_stage%d_task%d.jsonl'%(stage,t) for t in range(6,stage+1)]:
   assert not any(r['event']=='technical_exception' for r in jl(root/name))
 rates[seed]=rate
bytask=[];bytseed=[];nbt=[]
for task in range(6,10):
 old=sum(rates[s][k,task][0] for s in rates for k in range(task,10))/F(3*(10-task))
 new=sum(rates[s][k,task][1] for s in rates for k in range(task,10))/F(3*(10-task))
 diag=sum(rates[s][task,task][1]-rates[s][task,task][0] for s in rates)/3
 final=sum(rates[s][9,task][1]-rates[s][9,task][0] for s in rates)/3
 bytask.append(dict(task=task,parent_curve_mean=float(old*100),new_curve_mean=float(new*100),curve_delta=pack(new-old),auc_contribution=pack((new-old)/4),diagonal_delta=pack(diag),final_delta=pack(final)))
 if task<9:
  future=sum(rates[s][k,task][1]-rates[s][k,task][0] for s in rates for k in range(task+1,10))/F(3*(9-task))
  nbt.append(dict(task=task,diagonal_shift=pack(diag),future_mean_shift=pack(future),NBT_contribution=pack((diag-future)/3)))
 for seed in rates:
  contrib=sum(rates[seed][k,task][1]-rates[seed][k,task][0] for k in range(task,10))/F(4*(10-task))
  bytseed.append(dict(seed=seed,task=task,auc_contribution=pack(contrib)))
total=sum(F(v['auc_contribution']['fraction']) for v in bytask)
assert total==F(accepted['gates']['stage2']['mean_fraction'])
assert sum(F(v['NBT_contribution']['fraction']) for v in nbt)==F(-2,45)
ct=Counter(r['pair'] for r in pairs);assert len(pairs)==600
by_seed_pairs={str(s):dict(Counter(r['pair'] for r in pairs if r['seed']==s)) for s in rates}
by_task_pairs={str(t):dict(Counter(r['pair'] for r in pairs if r['task']==t)) for t in range(6,10)}
negative=[x for x in cells if x['delta_pp']<0];positive=[x for x in cells if x['delta_pp']>0]
diagonal_contrib=sum(F(str(x['delta_pp']))/100/F(4*(10-x['task']))/3 for x in cells if x['stage']==x['task'])
metrics=dict(AUC_delta=pack(total),diagonal_auc_contribution=pack(diagonal_contrib),later_auc_contribution=pack(total-diagonal_contrib),
             cells_negative=len(negative),cells_positive=len(positive),cells_zero=30-len(negative)-len(positive),
             descriptive_pairs=dict(ct),pairs_by_seed=by_seed_pairs,pairs_by_task=by_task_pairs,
             SF_retention=ct['SS']/(ct['SS']+ct['SF']),rescued_failure=ct['FS']/(ct['FF']+ct['FS']))
rng_states=js(OUT/'RNG_STATE_COMPARISON.json')
assert rng_states['status']=='PASS_READONLY' and rng_states['snapshots']==60 and not rng_states['cuda_initialized']
assert all(x['all_equal'] for x in rng_states['rows'] if x['task']==6)
summary=dict(status='PASS_READONLY_ANALYSIS',time=datetime.datetime.now().astimezone().isoformat(),
             evaluation_configs=configs,rng_states=dict(cells=30,equal_cells=rng_states['equal_cells'],
             all_stage_start_task6_equal=True,different=[dict(seed=x['seed'],stage=x['stage'],task=x['task']) for x in rng_states['rows'] if not x['all_equal']]),
             metrics=metrics,by_task=bytask,by_seed_task=bytseed,NBT_decomposition=nbt,calibration_epochs=epochs,
             structure=dict(checks=len(structs),max_off_gate=max(r['off_gate'] for r in structs),
                            max_self_error=max(r['self_gate_error'] for r in structs),
                            max_cross_contribution=max(r['cross_contribution_and_mean'] for r in structs),
                            max_delta=max(r['delta_max'] for r in structs),max_saturation=max(r['saturation_fraction'] for r in structs)),
             rng_budget=dict(groups=len(rng_groups),different_lengths=sum(x['parent_steps']!=x['new_steps'] for x in rng_groups),
                             different_prior_call_offsets=sum(not x['draw_budget_aligned'] for x in rng_groups),
                             caveat='Equal counts do not prove matching RNG states; nonisolated global RNG and no per-step noise trace prevent trajectory-level causal attribution'),
             costs=costs,limitations=['Training epoch loss is in-sample and epoch0 is already updating; no held-out calibration metric.',
                                     'Pairs match seed/stage/task/initial hash but are not isolated stochastic paired trials.',
                                     'No real observation/action trajectory or dimension-specific delta logs; cannot localize physical failure cause.',
                                     'No new training, model forward, checkpoint load or environment evaluation.'])
for n,v in [('ANALYSIS.json',summary),('CELL_DECOMPOSITION.json',cells),('PAIRED_OUTCOMES.json',pairs),('RNG_CALL_BUDGET.json',rng_groups)]:
 (OUT/n).write_text(json.dumps(v,indent=2)+'\n')
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in inputs.items())
(OUT/'INPUT_HASHES.json').write_text(json.dumps(dict(status='UNCHANGED',files=inputs),indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['metrics','by_task','NBT_decomposition','structure','rng_budget','calibration_epochs']},indent=2))
