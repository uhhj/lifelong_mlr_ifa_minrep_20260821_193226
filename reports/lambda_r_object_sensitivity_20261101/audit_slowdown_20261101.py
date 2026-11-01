"""Recompute read-only resource evidence; no remote actions or scientific updates."""
from pathlib import Path
import json, hashlib, statistics

R = Path(__file__).resolve().parent
D = R / 'capacity_readonly_20261101_182401'
raw = (D/'EVIDENCE.json').read_bytes()
e = json.loads(raw)
a,b = e['samples'][0],e['samples'][-1]
dt = b['time']-a['time']
def stat(s): return s[s.rindex(')')+2:].split()
def kv(s): return {k.rstrip(':'):int(v) for k,v in (line.split() for line in s.splitlines())}
x,y = stat(a['stat']),stat(b['stat'])
io0,io1 = kv(a['io']),kv(b['io'])
v0,v1 = kv(a['vmstat']),kv(b['vmstat'])
h0,h1 = [list(map(int,s['host_cpu'][0].split()[1:])) for s in (a,b)]
host = dict(zip(['user','nice','system','idle','iowait','irq','softirq','steal','guest','guest_nice'],[v-u for u,v in zip(h0,h1)]))
assert e['optimizer_calls']==e['environments']==e['remote_writes']==e['signals_sent']==0
assert e['profiler_attached'] is False
stages={}
for stage in (6,7):
    rows=[z for z in e['history'] if z['stage']==stage]
    expected=157 if stage==6 else 227
    assert all(z['batches']==expected for z in rows)
    previous_physical=0 if stage==6 else 15700
    previous_gates=0
    for z in rows:
        assert z['physical_optimizer_calls']-previous_physical==expected*(1 if stage==6 else 2)
        assert z['gate_blocks']-previous_gates==(0 if stage==6 else 23)
        previous_physical=z['physical_optimizer_calls'];previous_gates=z['gate_blocks']
    stages[str(stage)]={'epochs':len(rows),'ten_epoch_groups':[{'start':i+1,'count':len(rows[i:i+10]),'mean_seconds':statistics.mean(z['seconds'] for z in rows[i:i+10])} for i in range(0,len(rows),10)],'fixed_batch_count':expected,'fixed_physical_per_epoch':expected*(1 if stage==6 else 2)}
assert all(s['cgroups']['/sys/fs/cgroup/cpu/cpu.cfs_quota_us'].strip()=='-1' for s in e['samples'])
assert kv(b['cgroups']['/sys/fs/cgroup/cpu/cpu.stat'])['nr_throttled']==0
audit={
 'status':'PASS_READONLY_RESOURCE_RECOMPUTATION_NOT_ROOT_CAUSE_PROOF',
 'evidence_sha256':hashlib.sha256(raw).hexdigest(),'observed':e['observed'],'window_seconds':dt,
 'process_cpu_seconds':sum(int(y[i])-int(x[i]) for i in (11,12))/e['clock_ticks'],
 'main_thread_runqueue_wait_seconds':(int(b['schedstat'].split()[1])-int(a['schedstat'].split()[1]))/1e9,
 'visible_cpu_quota':'unlimited','visible_cpu_throttled_periods':0,'affinity_cpu_count':len(e['affinity']),
 'host_cpu_tick_deltas':host,'process_io_byte_deltas':{k:io1[k]-io0[k] for k in ['read_bytes','write_bytes','rchar','wchar']},
 'vm_deltas':{k:v1[k]-v0[k] for k in ['pgmajfault','pswpin','pswpout']},
 'io_full_pressure_seconds':(int(b['pressure']['io'].splitlines()[1].split('total=')[1])-int(a['pressure']['io'].splitlines()[1].split('total=')[1]))/1e6,
 'gpu_samples':[s['gpu']['stdout'].strip() for s in e['samples'] if 'gpu' in s],
 'stage_workload_and_timing':stages,
 'limits':['31-second sampling spans checkpoint activity; not a representative long-duration profile.',
 'No visible guest CPU quota/throttling or significant main-thread runqueue wait; hypervisor/provider constraints not completely observable.',
 'Host I/O stalls and checkpoint-sized process writes observed, but epoch seconds exclude subsequent checkpoint saves and the epoch-end probe; I/O alone does not explain intra-epoch slowdown.',
 'Low GPU utilization plus CPU activity is compatible with host-side computation/copy/synchronization; exact per-function cost not measured.',
 'Last four Task7 epochs are faster; cannot infer a stable recovery or a fixed completion date.'],
 'extra_optimizer_calls':0,'extra_environments':0,'production_modified':False,
 'progress_status':'PARTIAL; latest full progress audit remains check_readonly_20261101_181931'
}
(D/'LOCAL_SLOWDOWN_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
