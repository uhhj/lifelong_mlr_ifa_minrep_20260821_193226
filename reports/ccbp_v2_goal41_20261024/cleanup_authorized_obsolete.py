"""User-authorized removal of exactly three obsolete recovery files; no directories."""
import os,json,time,hashlib,subprocess
from pathlib import Path
receipt=Path('/data/mlr_ifa/experiment_workspaces/CCBP_FULL_REPLACEMENT_CLEANUP_20261024.json')
assert not receipt.exists(), 'Do not repeat cleanup; inspect existing receipt'
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1')
E=Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering05')
T=Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261024_recovery06')
targets=[('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering05/provenance/B0_task7_complete_recovery.pth','b7db9f59cbcec4fc5fdc523304917615ab9ed9af1320f14bb2618488c7b1b81f'),('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261024_recovery06/provenance/B2_task8_epoch90_recovery.pth','4925dba1a037bd2d106c139cd13a7b0f77f95fe755dea30ee171167978badf5b'),('/root/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1/B/recovery.pth','6b7c32c57955933a46ad210d9316c2849f6b672a7e5b2093ffd0f2f2ad192368')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert (Q/'controller.exit').read_text()=='1'
assert (T/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for d in Path('/proc').glob('[0-9]*'):
 try:cmd=(d/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 assert not any(str(q) in cmd for q in (Q,E,T)),(d.name,cmd)
# Completed CAET experiment's final models and current recovery files must remain available.
entries=json.loads((E/'ENTRY.json').read_text())
retained=[]
for arm in ('B0','B1','B2'):
 root=Path(entries['arms'][arm])
 for p in [root/'models'/('task%d.pth'%t) for t in range(6,10)]+[root/'recovery/latest.pth']:
  assert p.is_file() and p.stat().st_size>1000000
  retained.append(dict(path=str(p),bytes=p.stat().st_size))
rows=[]
for name,h in targets:
 p=Path(name);assert p.resolve()==p and p.is_file() and not p.is_symlink()
 st=p.stat();assert st.st_nlink==1 and sha(p)==h
 rows.append(dict(path=name,bytes=st.st_size,allocated=st.st_blocks*512,sha256=h,deleted=False))
result=dict(status='AUTHORIZED_EXACT_CLEANUP_PREPARED',time=time.time(),authorization='User: 可以自动删除一些废用的实验数据',rows=rows,retained_entry=entries,retained_final_assets=retained,scope='Only obsolete intermediate recovery states; no final model, raw data, logs, source or lightweight audit removed.',limitation='Historical recovery reenactment requiring these exact intermediate payloads is intentionally retired; historical completed audits remain preserved and are not claimed to be rerunnable after this authorized retirement.')
receipt.write_text(json.dumps(result,indent=2))
for row in rows:
 p=Path(row['path']);assert p.is_file() and sha(p)==row['sha256'];p.unlink();assert not p.exists();row['deleted']=True
 receipt.write_text(json.dumps(result,indent=2))
result['status']='EXACT_OBSOLETE_RECOVERY_CLEANUP_COMPLETE';result['after_capacity']={}
for mount in ['/root','/data']:
 v=os.statvfs(mount);result['after_capacity'][mount]=dict(bavail=v.f_bavail*v.f_frsize,bfree=v.f_bfree*v.f_frsize)
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result))
