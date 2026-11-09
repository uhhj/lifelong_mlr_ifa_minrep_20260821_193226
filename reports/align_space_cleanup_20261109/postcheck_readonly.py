from pathlib import Path
import os,json,hashlib,time
R=Path('/data/mlr_ifa/cleanup_20261109_align_space');receipt=json.loads((R/'RECEIPT.json').read_text());intent=json.loads((R/'INTENT.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert not Path(receipt['removed']['path']).exists()
assert all(sha(p)==h for p,h in intent['preserved_sha256'].items())
Q=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1');c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'S_Q1_' in cmd:processes.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
cache=[]
for p in Path('/data/mlr_ifa/cache').rglob('*'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size>1024**2:cache.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(dict(status='PASS_POST_CLEANUP_READONLY_VERIFICATION',observed=time.time(),receipt=receipt,preserved_file_hashes_verified=len(intent['preserved_sha256']),current_q1_frozen_files_exact=len(c['files']),current_q1_processes=processes,data_available=os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize,cache_large_files_preserved=cache,new_deletions=0,new_optimizer=0,new_environments=0)))
