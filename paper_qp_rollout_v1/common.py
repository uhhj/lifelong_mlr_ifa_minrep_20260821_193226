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
def trial_summary(done,steps):
    import numpy as np
    a=np.asarray(done,dtype=bool);ss=np.asarray(steps);assert a.shape[1]==10
    out=[]
    for lane in range(10):
        hit=ss[(ss>0)&a[:,lane]]
        out.append({'lane':lane,'success':bool(len(hit)),'first_success_step':int(hit[0]) if len(hit) else None})
    return out
