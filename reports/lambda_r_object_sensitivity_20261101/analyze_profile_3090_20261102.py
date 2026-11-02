"""Recompute bounded, nonblocking samples; do not interpret CPU IP shares as CUDA time."""
from pathlib import Path
import json,hashlib,collections,statistics
R=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=R/'capacity_readonly_20261102_142850/EVIDENCE.json'
windows=R/'capacity_readonly_20261102_143018/EVIDENCE.json'
symbols=R/'capacity_readonly_20261102_144037/EVIDENCE.json'
bench=R/'capacity_readonly_20261102_143242/EVIDENCE.json'
n=load(native);w=load(windows);sy=load(symbols);b=load(bench)
assert n['lost_samples']==0 and n['sample_count']==4338
assert b['status']=='PASS_MICROBENCH_EXACT_CHECKS_ONLY' and b['complete_state_unchanged']
profiles=[n['python_profile']]+w['windows'];counts=collections.Counter();sub=collections.Counter();leaves=collections.Counter();per=[];total=0
for profile in profiles:
    assert profile['exit']==0
    c=collections.Counter();nt=0
    for line in profile['stacks'].splitlines():
        stack,num=line.rsplit(' ',1);num=int(num)
        if ': MainThread;' not in stack:continue
        total+=num;nt+=num;leaves[stack.split(';')[-1]]+=num
        if '/updates.py:' in stack:
            category='candidate_update'
            if 'backward (' in stack:sub['backward']+=num
            elif '/updates.py:36' in stack:sub['finite_check']+=num
            elif '/optim/' in stack:sub['optimizer']+=num
            else:sub['other_update']+=num
        elif 'probe (' in stack:category='probe'
        elif 'restore (' in stack:category='restore'
        elif 'capture (' in stack:category='capture'
        elif 'equal (' in stack or '<lambda> (' in stack:category='equality'
        elif 'save (' in stack:category='save'
        elif '__next__ (' in stack:category='loader'
        else:category='other_or_partial'
        counts[category]+=num;c[category]+=num
    per.append(dict(samples=nt,counts=dict(c)))
assert total==1797
micro={}
for key,field,old,new in [('state_check_rounds','seconds','cpu','device'),('finite_check_rounds','seconds_per_call','serial','batched')]:
    x=[row[field] for row in b[key] if row['mode']==old];y=[row[field] for row in b[key] if row['mode']==new]
    assert len(x)==len(y)==12
    micro[key]=dict(reference_mean=statistics.mean(x),candidate_mean=statistics.mean(y),reference_median=statistics.median(x),candidate_median=statistics.median(y),pairs=12,faster_pairs=sum(v<u for u,v in zip(x,y)),fraction_saved=1-statistics.mean(y)/statistics.mean(x))
lower={name:sum(x['samples'] for x in sy['resolved_top_addresses'] if name in x['full_symbol']) for name in ['cpu_equal','direct_copy_kernel']}
out=dict(status='PASS_PROFILE_AND_MICROBENCH_RECOMPUTED_NOT_FULL_TRAINING_EQUIVALENCE',
    sources={str(p.relative_to(R)):sha(p) for p in [native,windows,symbols,bench]},
    python_main_samples=total,windows=per,categories={k:dict(samples=v,percent=100*v/total) for k,v in counts.items()},
    update_subcategories={k:dict(samples=v,percent_of_all=100*v/total) for k,v in sub.items()},
    top_python_leaves=leaves.most_common(20),native_active_main_thread_samples=n['sample_count'],native_lost=0,
    native_lower_bounds={k:dict(samples=v,percent_active_main_thread=100*v/n['sample_count']) for k,v in lower.items()},
    whitelisted_initial_environment=sy['whitelisted_initial_environment'],microbenchmark=micro,
    illustrative_total_time_saving_fraction=(counts['equality']/total*micro['state_check_rounds']['fraction_saved']+sub['finite_check']/total*micro['finite_check_rounds']['fraction_saved']),
    zero_optimizer_environments_signals_production_writes=True,
    caveats=['Three 60s 10Hz nonblocking Python windows; incomplete stack reads remain possible even without errors.',
      'CPU IP observations are active main-thread instruction shares, not GPU kernel timing or wall-time fractions.',
      'Named native loops are lower bounds among top100 saved addresses; unresolved CUDA driver addresses are not mislabeled.',
      'Microbenchmark shared GPU with production; fractional gains are only local check costs, not production speedup.',
      'Full paired training-path diagnostic is separate and must pass before any future production switch.'])
(R/'PROFILE_3090_LOCAL_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(samples=total,categories=out['categories'],native_lower_bounds=out['native_lower_bounds'],conditional_fraction=out['illustrative_total_time_saving_fraction']),ensure_ascii=False))
