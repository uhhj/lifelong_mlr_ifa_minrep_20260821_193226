"""Files/state helpers, kept outside historical implementations."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs/fixed_kappa_confirmation_v2'
REPORTS=ROOT/'reports/fixed_kappa_confirmation_v2'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)
def revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def wait_for(path,abort=None,timeout=24*3600):
    start=time.monotonic()
    while not Path(path).is_file():
        if abort and Path(abort).exists():raise RuntimeError('Queue integrity failure: '+str(abort))
        if time.monotonic()-start>timeout:raise TimeoutError(str(path))
        time.sleep(1)
def state_to_cpu(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:state_to_cpu(v) for k,v in x.items()}
    if isinstance(x,tuple):return tuple(state_to_cpu(v) for v in x)
    if isinstance(x,list):return [state_to_cpu(v) for v in x]
    return x
def rng_state():
    import random,numpy as np,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),
            'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def restore_rng(x):
    import random,numpy as np,torch
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch_cpu'])
    if x['torch_cuda']:torch.cuda.set_rng_state_all(x['torch_cuda'])
def exact(a,b,path='root'):
    import numpy as np,torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):raise AssertionError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+' keys')
        for k in a:exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError(path+' sequence')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
    elif a!=b:raise AssertionError(path+' scalar')
def atomic_torch_save(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)

def state_digest(x):
    import numpy as np,torch
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=repr):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode())
            for y in v:visit(y)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()

def environment():
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_EGL_DEVICE_ID='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
               EGL_PLATFORM='surfaceless',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1',
               PYTHONPATH=str(ROOT),HF_HOME=str(ROOT/'data/huggingface_cache'),
               LIBERO_CONFIG_PATH=str(ROOT/'data/libero_config'))
    compat=ROOT/'runtime/optimizer_signal_cuda580173/extracted/usr/lib/x86_64-linux-gnu'
    if compat.is_dir():env['LD_LIBRARY_PATH']=str(compat)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    graphics=ROOT/'runtime/confirmation_gl580173/extracted'
    if graphics.is_dir():
        env['LD_LIBRARY_PATH']=str(graphics/'usr/lib/x86_64-linux-gnu')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        env['__EGL_VENDOR_LIBRARY_FILENAMES']=str(graphics/'usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    return env
