"""Recompute external profile and exact-copy validation; does not run training."""
import json,hashlib,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sample=R/'capacity_readonly_20261029_093957/EVIDENCE.json'
validation=R/'capacity_readonly_20261029_094623/EVIDENCE.json'
v=js(sample);b=js(validation)
assert v['profiler_exit']==0 and v['frozen_files_exact'] and v['prefix_exact']
assert b['status']=='PASS_COPY_ONLY_EXACT_NO_TRAINING_REPLAY' and all(b[k] for k in ['dtype_shape_stride_value_exact','source_mutation_isolation','cpu_input_independence','rng_exact','frozen_files_exact'])
assert v['optimizer_calls_added_by_diagnostic']==b['extra_optimizer_calls']==b['extra_autograd_calls']==b['extra_environments']==0
counts=collections.Counter();n=0
for line in v['stacks'].splitlines():
    stack,count=line.rsplit(' ',1);count=int(count)
    if 'MainThread;' not in stack:continue
    n+=count
    if 'capture (' in stack:key='state_capture'
    elif 'restore (' in stack:key='state_restore'
    elif 'equal (' in stack:key='state_equality'
    elif 'probe (' in stack:key='fixed_probe'
    elif 'update (' in stack:key='training_update'
    elif 'save (' in stack or '_save (' in stack:key='checkpoint_save'
    elif '__next__ (' in stack or '__getitem__ (' in stack:key='data_load'
    else:key='other_or_incomplete'
    counts[key]+=count
first,last=v['samples'][0],v['samples'][-1];dt=last['time']-first['time']
fst=first['proc_stat'].split(')')[1].split();lst=last['proc_stat'].split(')')[1].split()
proc_cpu=100*sum(int(lst[i])-int(fst[i]) for i in (11,12))/v['clock_ticks']/dt
def io(s):return {k:int(val) for k,val in (x.split(':') for x in s.splitlines())}
ia,ib=io(first['io']),io(last['io'])
gpu=[float(x['gpu']['out'].split(',')[0]) for x in v['samples'] if x['gpu']['rc']==0]
med={k:statistics.median(vals) for k,vals in b['timings_seconds'].items()}
fraction=counts['state_capture']/n*(1-med['candidate']/med['original'])
out=dict(status='PASS_DIAGNOSTIC_COPY_PRIMITIVE_ONLY_NOT_DEPLOYED',main_thread_samples=n,category_samples=dict(counts),category_percent={k:100*x/n for k,x in counts.items()},duration_seconds=dt,process_cpu_percent_one_core=proc_cpu,gpu_utilization_percent=dict(mean=statistics.mean(gpu),minimum=min(gpu),maximum=max(gpu)),disk_read_bytes=ib['read_bytes']-ia['read_bytes'],disk_write_bytes=ib['write_bytes']-ia['write_bytes'],logical_read_bytes=ib['rchar']-ia['rchar'],scheduler_wait_seconds=(int(last['schedstat'].split()[1])-int(first['schedstat'].split()[1]))/1e9,cpu_quota=last['cpu_quota'].strip(),cpu_throttling=last['cpu_cgroup_stat'],copy_median_seconds=med,copy_speedup=med['original']/med['candidate'],conditional_whole_training_time_reduction_fraction=fraction,conditional_whole_training_speedup=1/(1-fraction),candidate='performance_copy_candidate.py',deployment='NOT_DEPLOYED',reason='Current interpreters have already imported the original function. The frozen controller starts fixed train.py paths, has no boundary hook, and forbids a second dispatch. Safe adoption would require a separate versioned queue handoff and full training admission; no hot patch, controller interruption, or replay was performed for this diagnostic.',added_optimizer_calls=0,added_autograd_calls=0,added_environment_attempts=0,limits=['Sample percentages are asynchronous wall-stack occupancy, not GPU kernel timings; about 5% other/incomplete frames retained.','Only a two-minute Task8 window was sampled; cannot attribute all historical slowdown or other stages with certainty.','Copy equivalence validates values/dtypes/strides, source isolation and RNG on a completed checkpoint in a separate process; it does not certify a modified whole training run.','Amdahl estimate assumes the microbenchmark reduction transfers to production; no production speedup claimed.'],sources={str(p.relative_to(R)):sha(p) for p in [sample,validation,R/'capacity_readonly_20261029_093718/EVIDENCE.json',R/'capacity_readonly_20261029_093843/EVIDENCE.json',R/'performance_copy_candidate.py',R/'controller.py',R/'seed42/train.py',R/'seed42/runtime.py']})
(R/'PERFORMANCE_AUDIT_20261029.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
