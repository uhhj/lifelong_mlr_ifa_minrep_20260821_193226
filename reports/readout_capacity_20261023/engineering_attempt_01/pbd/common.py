import os,sys,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['PBD_ROOT']);OUT=ROOT/'output'
OLD=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03')
os.environ['PEP_ROOT']=str(OLD);sys.path.insert(0,str(OLD))
from physical_effect_probe.common import sha,js,write,save,seed,full_policy,config,shape,PARENT
from physical_effect_probe.common import verify as verify_old
torch.set_num_threads(1)
V2=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2')
BUDGET=Path('/data/mlr_ifa/experiment_workspaces/READOUT_BUDGET_OBJECT41_20261023_v1')
SCALE=torch.tensor([.1]*6+[.08])
def verify():
 c=js(ROOT/'CONTRACT.json');bad=[p for p,h in c['files'].items() if sha(p)!=h];assert not bad,bad
 assert verify_old()==c['old_contract_sha256']
 assert sha(V2/'CONTRACT.json')==c['v2_contract_sha256']
 vc=js(V2/'CONTRACT.json');assert all(sha(p)==h for p,h in vc['files'].items())
 assert sha(BUDGET/'CONTRACT.json')==c['budget_contract_sha256']
 bc=js(BUDGET/'CONTRACT.json');assert all(sha(p)==h for p,h in bc['files'].items())
 return sha(ROOT/'CONTRACT.json')
def capacity(need=0):
 s=os.statvfs('/data');free=s.f_bfree*s.f_frsize;assert free>=2*1024**3+need,(free,need)
 s=os.statvfs('/');assert s.f_bavail*s.f_frsize>=1024**3
 return free
def log(event,**kw):
 OUT.mkdir(exist_ok=True,parents=True)
 with (OUT/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(event=event,unix=time.time(),**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def data(task):
 d=torch.load(OLD/'output/private'/('labels%d.pt'%task),map_location='cpu');x=torch.load(OLD/'output/private'/('native%d.pt'%task),map_location='cpu')
 assert sha(OLD/'output/private'/('labels%d.pt'%task))==js(OLD/'output/INPUT_BINDING_AUDIT.json')['tasks'][str(task)]['labels_sha256']
 assert sha(OLD/'output/private'/('native%d.pt'%task))==next(z['sha256'] for z in js(OLD/'output/NATIVE_VIEW_MANIFEST.json')['tasks'] if z['task']==task)
 return d,x
