import os,json,time,hashlib,subprocess
from pathlib import Path
N=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def verify_controller():
 for n,h in js(N/'AUTONOMOUS_CONTRACT.json')['files'].items():assert sha(N/n)==h,('controller code changed',n)
def phase(E,name,cmd,env=None):
 verify_controller();E.mkdir(parents=True,exist_ok=True);assert not (E/(name+'.exit')).exists() and not (E/(name+'.log')).exists(),('Already attempted',E,name)
 command=cmd if env is None else ['bash','-c','source '+str(env)+' && '+__import__('shlex').join(cmd)]
 with (E/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,command=command,time=time.time()))+'\n')
 start=time.time()
 with (E/(name+'.log')).open('x') as f:r=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 (E/(name+'.exit')).write_text(str(r.returncode)+'\n')
 with (E/'timings.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,exit=r.returncode,seconds=time.time()-start))+'\n')
 assert r.returncode==0,('phase failed',str(E),name,r.returncode)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
def no_workers():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();env=(p/'environ').read_bytes()
   if (b'RA_OBJECT_conditional_20261019' in env or b'RA_ZRv2_autonomous_20261019' in env) and any(x in cmd for x in ['-m coreset_exp.train','-m coreset_exp.evaluate','-m zr2_exp.calibrate','-m zr2_exp.evaluate','ra_adapter.py train']) and alive(p.name):found.append(cmd)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 assert not found,found
