import hashlib,json,os
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'reports/paper_qp_rollout_v1'
PRIOR=ROOT/'reports/paper_active_set_v1'
SCRATCH=Path('/root/paper_qp_rollout_v1_endpoints')
ARMS=['MLR','TASK_GRAD_QP']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def verify():
    for p,h in read(OUT/'FROZEN_INPUTS.json').items():assert sha(ROOT/p)==h,('frozen input changed',p)
def round_seed(point_index,task,round_index):return 9100000+point_index*100+task*2+round_index
def observation_digest(value):
    """Hash observation contents, never the pointers of NumPy object arrays."""
    import numpy as np
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,np.ndarray):
            h.update(repr((v.dtype.str,v.shape)).encode())
            if v.dtype.hasobject:
                for item in v.flat:visit(item)
            else:h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=repr):visit(k);visit(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:visit(x)
        else:h.update(repr((type(v).__name__,v)).encode())
    visit(value);return h.hexdigest()
def trial_summary(done,steps):
    import numpy as np
    a=np.asarray(done,dtype=bool);ss=np.asarray(steps);assert a.shape[1]==10
    out=[]
    for lane in range(10):
        hit=ss[(ss>0)&a[:,lane]]
        out.append({'lane':lane,'success':bool(len(hit)),'first_success_step':int(hit[0]) if len(hit) else None})
    return out
