"""Fresh-process parent numerical-mode admission; zero optimizer/environment calls."""
import os,sys,json,time,hashlib,copy
from pathlib import Path
P=Path('/data/mlr_ifa/analyses/Zero_response_residual_prototype_20261017')
O=Path('/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc')
W=Path(__file__).resolve().parent
os.environ['CORESET_OUT']=str(O)
sys.path.insert(0,str(W/'source'))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU
from coreset_exp.model import cfg,data_setup,complete
from coreset_exp.common import write,sha
from libero.lifelong.utils import control_seed

def main():
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ,'Prototype CUBLAS override leaked'
    assert not torch.are_deterministic_algorithms_enabled()
    control_seed(41)
    numeric=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic=torch.backends.cudnn.deterministic,algorithms=torch.are_deterministic_algorithms_enabled(),train_benchmark=torch.backends.cudnn.benchmark)
    c=cfg('eval');b,ds,shape=data_setup(c,True);c.shape_meta=shape
    p=complete(c,shape).eval().requires_grad_(False)
    gates=torch.from_numpy(np.load(P/'geometry/gates.npz')['gates']).cuda()
    choices=json.loads((P/'SAMPLE_IDS.json').read_text());rows=[]
    def check(e,t,kind,length):
        v=(e.double()/e.double().norm(dim=-1,keepdim=True))@gates.T
        old=[float(v[:,k-6].abs().max()) for k in range(max(6,t+1),10)]
        cur=float((v[:,t-6]-1).abs().max()) if t>=6 else 0.
        row=dict(task=t,kind=kind,length=length,batch=len(e),old_max=max(old,default=0),current_error=cur)
        row['passed']=row['old_max']<=1e-6 and cur<=1e-5;rows.append(row)
    with torch.no_grad():
        for t in range(6,10):
            d=torch.load(O/'features'/str(t)/choices['incremental'][str(t)]['probe'][0],map_location='cpu')
            check(d['feature'][0,:,0].cuda(),t,'cached',8)
        torch.backends.cudnn.benchmark=True
        for t in range(10):
            ids=choices['raw'][str(t)][:2]*5
            data=TU.map_tensor(next(iter(DataLoader(Subset(ds[t],ids),batch_size=10,num_workers=0))),lambda x:x.cuda())
            for length in (1,4,8):
                d=copy.deepcopy(data);d['obs']={k:v[:,:length].contiguous() for k,v in d['obs'].items()};d['actions']=d['actions'][:,:length]
                prepared=p.preprocess_input(d,train_mode=False);h=p.spatial_encode(prepared)
                check(h[:,-1,0],t,'native_10lane',length)
    result=dict(status='PASS' if all(r['passed'] for r in rows) else 'BLOCKED_NUMERICAL_COMPATIBILITY',numeric=numeric,eval_benchmark=torch.backends.cudnn.benchmark,rows=rows,optimizer_calls=0,environment_steps=0,gates_sha256=sha(P/'geometry/gates.npz'))
    write(W/'NUMERIC_ADMISSION.json',result);print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
