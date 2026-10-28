"""Read-only lightweight progress snapshot; never touches checkpoint payloads."""
import json,hashlib,os,subprocess,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1');P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
contract=js(Q/'CONTRACT.json');hashes={p:sha(p) for p in contract['files']};assert hashes==contract['files']
E=Q/'engineering_resume_20261028_v1'
engineering=None
if E.exists():
    ec=js(E/'CONTRACT.json');eh=sha(E/'CONTRACT.json')
    assert eh=='f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050'
    engineering_hashes={p:sha(E/p) for p in ec['files']}
    assert engineering_hashes==ec['files']
    engineering=dict(contract_sha256=eh,files=engineering_hashes)
light={};tails={};procs=[]
for p in Q.rglob('*'):
    if not p.is_file():continue
    n=p.relative_to(Q).as_posix()
    if p.suffix in ['.json','.jsonl','.exit'] and p.name not in ['BINDING.json','CASES.json','PREDEPLOY.json']:
        assert p.stat().st_size<12*1024**2
        raw=p.read_bytes();light[n]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
    elif p.suffix=='.log':tails[n]=p.read_text(errors='replace')[-3000:]
for p in Path('/proc').glob('[0-9]*'):
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:continue
    if str(Q) in cmd and int(p.name)!=os.getpid():procs.append(dict(pid=int(p.name),command=cmd))
capacity={};seen=set();allocated=0
for p in P.rglob('*') if P.exists() else []:
    if p.is_file() and not p.is_symlink():
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen.add(key);allocated+=st.st_blocks*512
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',max(0,8*1024**3+(346305333 if engineering else 0)-allocated),2*1024**3,True)]:
    st=os.statvfs(mount);free=(st.f_bfree if root else st.f_bavail)*st.f_frsize;capacity[mount]=dict(free=free,remaining_bound=bound,reserve=reserve,spare=free-bound-reserve)
gpu_result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
gpu_diagnostic=dict(exit=gpu_result.returncode,stdout=gpu_result.stdout,stderr=gpu_result.stderr)
perf=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
performance=dict(gpu_exit=perf.returncode,gpu=perf.stdout,gpu_error=perf.stderr,loadavg=Path('/proc/loadavg').read_text(),cpu_count=os.cpu_count())
print(json.dumps(dict(status='READONLY_SNAPSHOT_NOT_FINAL',observed=time.time(),contract=sha(Q/'CONTRACT.json'),frozen=hashes,engineering=engineering,processes=procs,capacity=capacity,light=light,log_tails=tails,gpu=gpu_result.stdout,gpu_diagnostic=gpu_diagnostic,performance=performance,host_uptime=Path('/proc/uptime').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
