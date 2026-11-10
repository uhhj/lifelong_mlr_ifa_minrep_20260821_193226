from pathlib import Path
import sys,json,hashlib,time,os,subprocess
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(Q))
import runtime as original
P=original.P;PY=original.PY;CASES=original.CASES
js=original.js;sha=original.sha;write=original.write;env=original.env;processes=original.processes
def event(kind,**kw):
 with (E/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n');f.flush();os.fsync(f.fileno())
def verify():
 original.verify();c=js(E/'CONTRACT.json')
 for n,h in c['files'].items():assert sha(E/n)==h,n
 for n,h in js(E/'PRESERVED.json')['files'].items():assert sha(n)==h,n
 assets=js(E/'ASSETS.json')
 for rel,h in assets['manifest'].items():assert sha(Path(assets['root'])/rel)==h,rel
 for case in CASES:assert (Q/'cases'/case/'source/libero/libero/assets').resolve()==Path(assets['root'])
 assert (Q/'controller.exit').read_text().strip()=='1'
 assert (P/'OBJECT_41/evaluate.exit').read_text().strip()=='1'
 return c
