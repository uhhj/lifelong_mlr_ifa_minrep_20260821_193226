import csv,hashlib,json,os
from pathlib import Path
ROOT=Path.cwd();OUT=ROOT/'reports/sample_task_qp_v1'
OLD=ROOT/'reports/paper_active_set_v1'
GEO=ROOT/'reports/functional_geometry_diagnostic_v1'
ARMS=['MLR','TASK_GRAD_QP','SAMPLE_GRAD_QP','TASK_STEP_QP','SAMPLE_STEP_QP']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(str(tmp),str(p))
def emit(p,obj):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def csvout(p,data):
    if not data:return
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
