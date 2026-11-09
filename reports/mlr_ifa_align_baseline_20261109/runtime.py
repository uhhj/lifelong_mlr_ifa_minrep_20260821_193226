"""Finite six-run orchestration helpers. No automatic retries or old-run writes."""
import hashlib,json,os,time,subprocess
from pathlib import Path
NAME='MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME
P=Path('/data/mlr_ifa/experiments')/NAME
PY='/data/mlr_ifa/env/bin/python'
OLD=Q.parent/'S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1'
OLD_HASH='fbbce2917380d1cde1025f0006b6fdd6bc7959f92226bb5588cc431ee0c8d96c'
CASES=[s+'_'+str(i) for s in ['OBJECT','GOAL'] for i in [41,42,43]]
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def event(kind,**kw):
 with (Q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n');f.flush();os.fsync(f.fileno())
def processes(roots):
 out=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if int(p.name)==os.getpid():continue
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   env=(p/'environ').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if any(str(r) in cmd or str(r) in env for r in roots):out.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
  except (OSError,ProcessLookupError):pass
 return out
def verify():
 c=js(Q/'CONTRACT.json')
 for rel,h in c['code_files'].items():assert sha(Q/rel)==h,rel
 for rel,h in c['private_files'].items():assert sha(P/rel)==h,rel
 return c
def previous_finished():
 assert sha(OLD/'CONTRACT.json')==OLD_HASH
 assert all(sha(path)==h for path,h in js(OLD/'CONTRACT.json')['files'].items())
 assert (OLD/'controller.exit').read_text().strip()=='0'
 roots=[OLD]
 for suite in ['OBJECT','GOAL']:
  root=OLD.parent/('S_Q1_'+suite+'41_42_43_20261109_rerun_v1');roots.append(root)
  assert (root/'controller.exit').read_text().strip()=='0'
  assert js(root/'Q1_EVALUATION_COMPLETE.json')['episodes']==600
  for seed in [41,42,43]:
   d=root/('Q1_'+str(seed));assert (d/'train_S.exit').read_text().strip()=='0' and (d/'evaluate.exit').read_text().strip()=='0'
 assert not processes(roots),'Current q=1 still active; never interrupt it'
def acceptance_gate():
 previous_finished();a=js(Q/'POST_Q1_ACCEPTANCE.json')
 assert a['status']=='PASS_BOTH_Q1_SUITES_ACCEPTED_AND_PUBLISHED'
 assert a['global_contract']==OLD_HASH and a['global_root']==str(OLD)
 for suite in ['OBJECT','GOAL']:
  row=a['suites'][suite];assert row['audit']['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
  assert row['audit']['contract']==sha(OLD.parent/('S_Q1_'+suite+'41_42_43_20261109_rerun_v1')/'CONTRACT.json')
  assert row['audit']['formal_episodes']==600 and row['audit']['epochs']==1200
  assert hashlib.sha256(row['audit_text'].encode()).hexdigest()==row['audit_sha256']
  assert json.loads(row['audit_text'])==row['audit']
  assert row['publication_path'] in a['published_files']
  assert a['published_files'][row['publication_path']]['sha256']==row['audit_sha256']
 assert len(a['remote_commit'])==40
def capacity():
 allocated=0;seen=set()
 if P.exists():
  for p in P.rglob('*'):
   if not p.is_file() or p.is_symlink():continue
   st=p.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);allocated+=st.st_blocks*512
 result={}
 for mount,bound,reserve in [('/data',6*1024**3,2*1024**3),('/root',128*1024**2,1024**3)]:
  s=os.statvfs(mount);free=s.f_bavail*s.f_frsize;remaining=max(0,bound-allocated) if mount=='/data' else bound
  result[mount]=dict(free=free,remaining=remaining,reserve=reserve,spare=free-remaining-reserve)
  assert free-remaining>=reserve,('Fresh capacity insufficient; no deletion or paid expansion',result)
 return result
def env(case):
 e=os.environ.copy();e.update(CORESET_OUT=str(P/case),CORESET_SEED=case[-2:],PYTHONPATH=str(Q/'cases'/case/'source'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
 return e
