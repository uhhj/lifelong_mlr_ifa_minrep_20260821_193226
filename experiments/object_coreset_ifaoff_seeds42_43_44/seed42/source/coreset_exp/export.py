import json,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import O,sha,write,verify,hashes
from .model import cfg,data_setup,complete
from libero.lifelong.algos.basic import Basic
from libero.lifelong.utils import control_seed

def main():
    verify();assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
    assert not list((O/'features').rglob('*.pt')),'Partial export: refuse blind overwrite'
    assert not (O/'FEATURES.json').exists()
    c=cfg('export');control_seed(42);b,ds,shape=data_setup(c,True);c.shape_meta=shape
    write(O/'resolved_export_actual.json',dict(c));control_seed(42)
    dist.init_process_group('nccl');torch.cuda.set_device(0);begin=time.time()
    try:
        a=Basic(10,c).cuda();strict=complete(c,shape) # explicit equality audit without changing formal RNG below
        del strict
        # Rebuild from exactly reset seed after the loading audit, like released exporter.
        del a;control_seed(42);a=Basic(10,c).cuda();a.policy=DDP(a.policy,device_ids=[0],output_device=0);a.train()
        counts={};demos={}
        for t in range(6,10):
            write(O/'RUN_STATUS.json',dict(status='EXPORTING',task=t,updated_unix=time.time(),pid=__import__('os').getpid()))
            a.learn_one_task(ds[t],t,b,{},save_feature=True,saving_dir=str(O/'features'))
            paths=list((O/'features'/str(t)).glob('*.pt'));assert len(paths)==len(ds[t])
            demos[str(t)]=list(ds[t].sequence_dataset.demos);counts[str(t)]=len(paths)
            for p in paths:
                x=torch.load(p,map_location='cpu');assert x['feature'].shape==(1,8,4,512) and x['actions'].shape==(1,8,7)
                assert torch.isfinite(x['feature']).all() and torch.isfinite(x['actions']).all()
        write(O/'FEATURES.json',dict(counts=counts,demos=demos,hashes=hashes(O/'features'),
            modal_sha256=sha(O/'MODAL_CONTRACT.json'),seconds=time.time()-begin,
            contract_sha256=sha(O/'CONTRACT.json'),export_mode='native train mode with original augmentation'))
        write(O/'RUN_STATUS.json',dict(status='FEATURES_READY',counts=counts,updated_unix=time.time()))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
