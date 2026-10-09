"""Read-only diagnostics of existing E1 selection logs; no torch/model/environment."""
import collections,csv,hashlib,json,math,statistics,time
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'20261008_E1_protective_selection'
E=ROOT/'final_20261009/evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
inputs=sorted((E/'selection_logs').rglob('*.jsonl'))+[E/('catalog_task%d.json'%t) for t in range(6,10)]+[E/'TRAINING_COMPLETE.json',E/'CONTRACT.json',ROOT/'PUBLICATION_RECEIPT.json']
before={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
assert len(list((E/'selection_logs').rglob('*.jsonl')))==400
totals=read(E/'TRAINING_COMPLETE.json')['selection_totals'];pools=[];empty_task6=0;batch_collisions=[]
for stage,batches in [(6,157),(7,227),(8,309),(9,362)]:
    cat=read(E/('catalog_task%d.json'%stage));members=collections.defaultdict(set)
    for key,row in cat.items():
        if key.startswith('buffer/'):members[row['task']].add(key)
    active={};stage_events=0
    for path in sorted((E/'selection_logs'/('task%d'%stage)).glob('epoch*.jsonl')):
        for line in path.read_text().splitlines():
            e=json.loads(line);stage_events+=1
            assert e['stage_optimizer_step']==stage_events and e['stage']==stage
            step=stage_events-1;r=e['refresh']
            if r:
                assert r['step']==step and step%500==0 and r['reason']=='OK'
                for task,ids in r['valid_ids'].items():
                    task=int(task);assert len(ids)==64 and len(set(ids))==64 and set(ids)<=members[task]
                    scores=np.asarray(r['scores'][str(task)],dtype=float);q=np.asarray(r['probabilities'][str(task)],dtype=float)
                    expected=np.exp(scores-scores.max());expected/=expected.sum()
                    assert np.allclose(expected,q,rtol=0,atol=1e-12)
                    order=np.argsort(-scores,kind='stable');top=set(order[:16]);bottom=set(order[-16:])
                    pool=dict(stage=stage,task=task,start=step,end=min(step+500,100*batches),ids=ids,index={k:i for i,k in enumerate(ids)},scores=scores,q=q,top=top,bottom=bottom,M=len(members[task]),draws=[])
                    pools.append(pool);active[task]=pool
            if stage==6:
                assert not r and not e['replacements'];empty_task6+=1
            per_task=collections.defaultdict(list)
            for x in e['replacements']:
                assert x['applied'] and x['task']<stage
                pool=active[x['task']];idx=pool['index'][x['candidate']]
                pool['draws'].append((step,x['slot'],idx));per_task[x['task']].append(idx)
            for task,idxs in per_task.items():
                n=len(idxs);c=collections.Counter(idxs);pool=active[task]
                if n>1:batch_collisions.append(dict(stage=stage,task=task,pairs=n*(n-1)/2,collisions=sum(v*(v-1)/2 for v in c.values()),q_collision=float(np.sum(pool['q']**2)),full_uniform=1/pool['M']))
    assert stage_events==100*batches
    assert sum(len(p['draws']) for p in pools if p['stage']==stage)==totals[str(stage)]['assigned_slots']
assert len(pools)==389 and empty_task6==15700

def pool_row(p,horizon=None):
    d=p['draws'] if horizon is None else [x for x in p['draws'] if x[0]<p['start']+horizon]
    n=len(d);idx=[x[2] for x in d];cnt=np.bincount(idx,minlength=64);q=p['q'];scores=p['scores'];M=p['M']
    eq=float(q@scores);unif=float(scores.mean());topq=sum(q[i] for i in p['top']);bottomq=sum(q[i] for i in p['bottom'])
    unique=int(np.count_nonzero(cnt));q_unique=float(np.sum(-np.expm1(n*np.log1p(-q))))
    u64=64*(-math.expm1(n*math.log1p(-1/64)));uM=M*(-math.expm1(n*math.log1p(-1/M)))
    recent={};lags=[]
    for step,slot,i in d:
        if i in recent:lags.append(step-recent[i])
        recent[i]=step
    return dict(stage=p['stage'],old_task=p['task'],refresh_step=p['start'],window_steps=min(p['end']-p['start'],horizon or 500),
        pool_size=64,buffer_size=M,draws=n,unique=unique,repeat_assignments=n-unique,
        uniform_pool_expected_unique=u64,weighted_pool_expected_unique=q_unique,full_buffer_expected_unique=uM,
        score_mean=unif,score_sd=float(scores.std()),score_min=float(scores.min()),score_max=float(scores.max()),
        score_span=float(np.ptp(scores)),q_min=float(q.min()),q_max=float(q.max()),q_max_min_ratio=float(q.max()/q.min()),
        q_total_variation_from_uniform=float(np.abs(q-1/64).sum()/2),q_ess=float(1/(q@q)),q_entropy_normalized=float(-np.sum(q*np.log(q))/math.log(64)),
        predicted_mean_selected_score=eq,observed_score_sum=float(cnt@scores),uniform_score_sum=n*unif,weighted_score_sum=n*eq,
        score_variance_sum=n*float(q@((scores-eq)**2)),top16_draws=int(sum(cnt[i] for i in p['top'])),
        bottom16_draws=int(sum(cnt[i] for i in p['bottom'])),top16_expected=n*float(topq),bottom16_expected=n*float(bottomq),
        top16_variance=n*float(topq*(1-topq)),q_top16_mass=float(topq),q_bottom16_mass=float(bottomq),
        maximum_count=int(cnt.max()),top8_observed_count=int(np.sort(cnt)[-8:].sum()),
        ordered_pairs=n*(n-1),equal_id_ordered_pairs=int(sum(int(x)*(int(x)-1) for x in cnt)),
        q_pair_probability=float(q@q),uniform_pool_pair_probability=1/64,uniform_buffer_pair_probability=1/M,
        mixture_half_uniform_half_priority_pair_probability=.75/M+.25*float(q@q),
        mixture_expected_mass_in_candidate_pool=.5+.5*64/M,
        adjacent_equal=sum(a[2]==b[2] for a,b in zip(d,d[1:])),adjacent_count=max(0,n-1),
        revisit_count=len(lags),revisit_within10=sum(x<=10 for x in lags),revisit_within50=sum(x<=50 for x in lags),
        revisit_lag_median=float(np.median(lags)) if lags else None)
rows=[pool_row(p) for p in pools]
def summarize(rs):
    N=sum(r['draws'] for r in rs);pairs=sum(r['ordered_pairs'] for r in rs);adj=sum(r['adjacent_count'] for r in rs)
    weighted=lambda k:sum(r['draws']*r[k] for r in rs)/N
    summ=lambda k:sum(r[k] for r in rs)
    qpair=sum(r['ordered_pairs']*r['q_pair_probability'] for r in rs)/pairs
    upair=sum(r['ordered_pairs']*r['uniform_buffer_pair_probability'] for r in rs)/pairs
    s=dict(pools=len(rs),draws=N,unique_pool_id_pairs=summ('unique'),repeat_fraction=1-summ('unique')/N,
        repeat_fraction_expected_weighted_pool=1-summ('weighted_pool_expected_unique')/N,
        repeat_fraction_expected_uniform_pool=1-summ('uniform_pool_expected_unique')/N,
        repeat_fraction_expected_full_buffer=1-summ('full_buffer_expected_unique')/N,
        q_ess_mean=weighted('q_ess'),q_entropy_mean=weighted('q_entropy_normalized'),q_tv_mean=weighted('q_total_variation_from_uniform'),
        score_sd_mean=weighted('score_sd'),score_span_mean=weighted('score_span'),q_max_min_mean=weighted('q_max_min_ratio'),
        q_top16_mass=summ('top16_expected')/N,actual_top16_fraction=summ('top16_draws')/N,
        q_bottom16_mass=summ('bottom16_expected')/N,actual_bottom16_fraction=summ('bottom16_draws')/N,
        uniform_mean_score=summ('uniform_score_sum')/N,predicted_selected_mean_score=summ('weighted_score_sum')/N,
        actual_selected_mean_score=summ('observed_score_sum')/N,
        selected_score_residual_se=math.sqrt(summ('score_variance_sum'))/N,
        top16_calibration_residual_z=(summ('top16_draws')-summ('top16_expected'))/math.sqrt(summ('top16_variance')),
        actual_pair_collision=summ('equal_id_ordered_pairs')/pairs,q_pair_collision=qpair,
        uniform_pool_pair_collision=1/64,full_buffer_pair_collision=upair,
        score_weighting_collision_factor=qpair/(1/64),pool_restriction_collision_factor=(1/64)/upair,
        actual_collision_factor_vs_full_buffer=(summ('equal_id_ordered_pairs')/pairs)/upair,
        observed_adjacent_repeat=summ('adjacent_equal')/adj,
        expected_adjacent_repeat=sum(r['adjacent_count']*r['q_pair_probability'] for r in rs)/adj,
        top8_observed_share=summ('top8_observed_count')/N,maximum_single_id_count=max(r['maximum_count'] for r in rs),
        revisit_within10_fraction=summ('revisit_within10')/summ('revisit_count'),
        revisit_within50_fraction=summ('revisit_within50')/summ('revisit_count'),
        mixture_model_effective_size=1/weighted('mixture_half_uniform_half_priority_pair_probability'),
        mixture_model_candidate_mass=weighted('mixture_expected_mass_in_candidate_pool'))
    return s
summary={f'stage{s}_old{t}':summarize([r for r in rows if r['stage']==s and r['old_task']==t]) for s in (7,8,9) for t in range(6,s)}
summary['all']=summarize(rows)
horizons={str(h):summarize([pool_row(p,h) for p in pools if p['end']-p['start']>=h]) for h in (50,100,500)}
transitions=[]
for s in (7,8,9):
    for t in range(6,s):
        ps=[p for p in pools if p['stage']==s and p['task']==t]
        for a,b in zip(ps,ps[1:]):
            A=set(a['ids']);B=set(b['ids']);overlap=A&B
            transitions.append(dict(stage=s,old_task=t,previous_refresh=a['start'],next_refresh=b['start'],overlap=len(overlap),
                expected_independent_uniform_overlap=64*64/a['M'],jaccard=len(overlap)/len(A|B)))
batch_summary={}
for s in (7,8,9):
    rs=[r for r in batch_collisions if r['stage']==s];den=sum(r['pairs'] for r in rs)
    batch_summary[s]=dict(priority_same_task_pairs=den,observed_equal_id_pair_rate=sum(r['collisions'] for r in rs)/den,
        expected_weighted_pool_rate=sum(r['pairs']*r['q_collision'] for r in rs)/den,
        hypothetical_full_buffer_uniform_rate=sum(r['pairs']*r['full_uniform'] for r in rs)/den)
diagnosis=dict(status='PASS',analysis_unix=time.time(),source_publication=read(ROOT/'PUBLICATION_RECEIPT.json')['commit'],
    scope='Existing priority-selection events only; no training, model, environment, new rollouts, random simulation or parameter changes.',
    priority_draws=sum(len(p['draws']) for p in pools),refresh_events=181,task_specific_candidate_pools=len(pools),
    summary=summary,horizons=horizons,batch_collisions=batch_summary,
    overlap_mean=float(np.mean([r['overlap'] for r in transitions])),overlap_expected_mean=float(np.mean([r['expected_independent_uniform_overlap'] for r in transitions])),
    pool_quantiles={k:dict(zip(['min','p10','median','p90','max'],[float(x) for x in np.quantile([r[k] for r in rows],[0,.1,.5,.9,1])])) for k in ['q_ess','q_top16_mass','score_span','q_max_min_ratio','maximum_count']},
    interpretation_limits=['Effective preference means a measurable shift in retrieval probability and selected proxy scores, not beneficial gradients or improved closed-loop behavior.',
      'Uniform same-pool and full-buffer numbers are analytic reference expectations conditional on observed draw counts, not newly executed control experiments.',
      'The full-buffer analytic reference is uniform sampling WITH replacement. E0 used a concatenated shuffled DataLoader, so this reference is not a reconstruction or measured E0 trajectory.',
      'Only priority assignments are logged with IDs. Kept-original IDs and full task-slot lists are absent; whole replay-stream unique coverage, actual changed-ID rate, and interactions with retained samples cannot be measured.',
      'The 50/50 whole-replay mixture is a mathematical probability model, not an observed count of the whole replay stream.',
      'Scores are frozen within each pool. Logs lack per-step rescoring and separate history/conflict gradients; stale ranking and gradient effectiveness cannot be established.',
      'Window IDs need not be independent observations or trajectories; no statistical/causal claims about closed-loop outcomes.'])
after={p.relative_to(ROOT).as_posix():sha(p) for p in inputs};assert after==before
original_manifest=read(ROOT/'final_20261009/FINAL_MANIFEST.json')
matched=0
for name,h in before.items():
    if name.startswith('final_20261009/'):
        assert original_manifest[name[len('final_20261009/'):]]==h,name
        matched+=1
assert matched==406
diagnosis['original_final_manifest_matches']=matched
write('INPUT_HASHES.json',dict(before=before,after=after,unchanged=True))
write('SELECTION_DIAGNOSTICS.json',diagnosis)
for name,rs in [('per_pool.csv',rows),('refresh_overlap.csv',transitions)]:
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
print(json.dumps(dict(all=summary['all'],groups={k:{x:v[x] for x in ['draws','q_ess_mean','actual_top16_fraction','repeat_fraction','repeat_fraction_expected_full_buffer','actual_collision_factor_vs_full_buffer']} for k,v in summary.items() if k!='all'},horizons={k:{x:v[x] for x in ['draws','repeat_fraction','repeat_fraction_expected_uniform_pool','repeat_fraction_expected_full_buffer']} for k,v in horizons.items()}),indent=2))
