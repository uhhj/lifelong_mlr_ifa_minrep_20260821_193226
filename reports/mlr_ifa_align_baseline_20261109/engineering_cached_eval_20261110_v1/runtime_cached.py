from pathlib import Path
import os,sys,json,hashlib,time,subprocess
E=Path(__file__).resolve().parent;Q=E.parent;OLD=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(OLD));import resume_runtime as prior
P=prior.P;PY=prior.PY;CASES=prior.CASES;env=prior.env;js=prior.js;write=prior.write;sha=prior.sha;processes=prior.processes
def verify():
    prior.verify();c=js(E/'CONTRACT.json')
    assert all(sha(E/n)==h for n,h in c['files'].items())
    if (E/'PRESERVED.json').exists():assert all(sha(Path(n))==h for n,h in js(E/'PRESERVED.json')['files'].items())
    return c
def event(kind,**fields):
    with (E/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**fields))+'\n');f.flush();os.fsync(f.fileno())
def proc(pid):
    f=Path('/proc')/str(pid)/'stat'
    if not f.exists():return None
    v=f.read_text().split(') ',1)[1].split();return dict(pid=pid,state=v[0],start=v[19],exit_code=int(v[49]))
def same(identity):
    p=proc(identity['pid']);assert p and p['start']==identity['start'],'Process identity changed';return p
