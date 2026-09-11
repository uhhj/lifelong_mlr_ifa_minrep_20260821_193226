"""NCCL lifecycle test, not a training or rollout experiment."""
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from confirmation_experiment.common import write

if __name__=='__main__':
    folder=Path(os.environ['ISOLATION_TEST_DIR']);role=os.environ['ISOLATION_TEST_ROLE']
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    x=torch.tensor([3.],device='cuda');dist.all_reduce(x)
    write(folder/(role+'_ready.json'),{'master_port':os.environ['MASTER_PORT'],'run_id':os.environ['TORCHELASTIC_RUN_ID'],'value':x.item()})
    if role=='keeper':
        deadline=time.monotonic()+150
        while not (folder/'probe_after_owner_exit').exists():
            if time.monotonic()>deadline:raise TimeoutError('owner exit signal')
            time.sleep(.2)
        time.sleep(12)  # Covers elastic monitoring/heartbeat after peer exit.
        for _ in range(5):dist.all_reduce(x)
        write(folder/'keeper_survived.json',{'status':'PASS','value':x.item()})
    else:
        deadline=time.monotonic()+150
        while not (folder/'owner_release').exists():
            if time.monotonic()>deadline:raise TimeoutError('owner release')
            time.sleep(.2)
    dist.destroy_process_group()
