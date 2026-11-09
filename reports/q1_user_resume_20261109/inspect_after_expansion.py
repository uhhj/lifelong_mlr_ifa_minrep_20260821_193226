import os,json,time,hashlib,subprocess
from pathlib import Path
M=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1')
d=json.loads((M/'user_pause_20261109_v1/STATUS.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
live={}
for pid,x in d['after'].items():
 p=Path('/proc')/pid
 if p.exists():
  s=(p/'stat').read_text().split(') ',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');live[pid]=dict(state=s[0],start=s[19],cmd=cmd,same_identity=s[19]==x['start'] and cmd==x['cmd'])
 else:live[pid]=dict(missing=True)
frozen=json.loads((M/'CONTRACT.json').read_text())['files']
changed=[p for p,h in frozen.items() if not Path(p).is_file() or sha(p)!=h]
light_changed=[p for p,h in d['light_hashes'].items() if not Path(p).is_file() or sha(p)!=h]
capacity={}
for m in ['/','/data']:
 v=os.statvfs(m);capacity[m]=dict(total=v.f_blocks*v.f_frsize,available=v.f_bavail*v.f_frsize,free_including_reserved=v.f_bfree*v.f_frsize)
allocated=0
for name in ['S_Q1_OBJECT41_42_43_20261109_rerun_v1','S_Q1_GOAL41_42_43_20261109_rerun_v1']:
 q=Path('/data/mlr_ifa/experiments')/name
 for p in q.rglob('*'):
  if p.is_file() and not p.is_symlink():allocated+=p.stat().st_blocks*512
commands={}
for k,c in [('mounts',['findmnt','-J','-T','/data']),('gpu',['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'])]:
 r=subprocess.run(c,capture_output=True,text=True);commands[k]=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(dict(observed=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uptime=Path('/proc/uptime').read_text().strip(),processes=live,frozen_count=len(frozen),frozen_changed=changed,paused_light_changed=light_changed,capacity=capacity,q1_allocated=allocated,q1_remaining_bound=max(0,6*1024**3-allocated),commands=commands)))
