"""Actual SC payload/selection and four isolated native optimizer calls; zero env steps."""
import os,json,time,pickle,hashlib
from pathlib import Path
import numpy as np,torch,h5py
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,sha,write,save,verify,rng
from coreset_exp.model import cfg,angular_mean
from coreset_exp.train import build,capture,restore,cpu
from ra_exp.storage import select
from ra_exp.validation import same_witness

def main():
    verify();assert not (O/'GPU_PREFLIGHT.json').exists();assert torch.cuda.is_available();torch.set_num_threads(4)
    begin=time.time();P=Path(os.environ['RA_PARENT']);fm=json.loads((O/'FEATURES.json').read_text());tasks=json.loads((O/'TASKS.json').read_text());out={}
    for t in range(6,10):
        paths=[str(p) for p in sorted((O/'features'/str(t)).glob('*.pt'))]
        with h5py.File(tasks[str(t)]['paths'][0]) as f:population=np.concatenate([f['data'][d]['actions'][:] for d in fm['demos'][str(t)]])
        binding=json.loads((O/'bindings'/f'task{t}.json').read_text())['windows']
        before=rng();chosen,report=select(paths,(len(paths)+1)//2,population,binding);after=rng()
        assert before['python']==after['python'] and torch.equal(before['cpu'],after['cpu']) and all(torch.equal(a,b) for a,b in zip(before['cuda'],after['cuda']))
        assert pickle.dumps(before['numpy'])==pickle.dumps(after['numpy'])
        parent=json.loads((P/'coreset/storage'/f'task{t}.json').read_text())
        assert report['original_ids']==[Path(r['source']).name for r in parent['selected']]
        offline=json.loads((O/'offline_selection'/f'task{t}.json').read_text())
        assert report['selected_ids']==offline['selected_ids']
        assert report['original_ids']==offline['original_ids']
        assert report['anchor_ids']==offline['anchor_ids'] and report['random_draw_ids']==offline['random_draw_ids']
        for key in ['max_before','max_after','mean_before','mean_after']:assert report[key]==offline[key]
        same_witness(report['swaps'],offline['swaps'])
        write(O/'selection_preflight'/f'task{t}.json',report)
        out[str(t)]=dict(N=len(paths),K=len(chosen),swaps=len(report['swaps']),original_matches_parent=True,offline_exact=True)
        print(json.dumps(dict(task=t,**out[str(t)])),flush=True)
    # Native latent batch and unchanged BC+IFA updates; clone updates are discarded.
    payload=[]
    for t in (6,7):
        for p in sorted((O/'features'/str(t)).glob('*.pt'))[:5]:payload.append((torch.load(p,map_location='cpu'),str(t)))
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        c=cfg('ra_coreset');c.run_root=str(O/'smoke');c.experiment_dir=str(O/'smoke/models');c.buffer_dir=str(O/'smoke/buffer')
        a,_=build(c);loader=DataLoader(payload,batch_size=10,num_workers=0);a.start_task(-1,loader)
        a.refs={6:payload[0][0]['feature'][0,0,0].cuda(),7:payload[5][0]['feature'][0,0,0].cuda()};info={'selected_pairs':[(6,7)]}
        a.observe(next(iter(loader)),7,0,info);a.observe(next(iter(loader)),7,0,info);state=capture(a);save(O/'smoke/recovery.pth',state)
        x=a.observe(next(iter(loader)),7,0,info);expected=capture(a);restore(a,torch.load(O/'smoke/recovery.pth',map_location='cpu'));y=a.observe(next(iter(loader)),7,0,info)
        assert x==y and all(torch.equal(v,expected['model'][k]) for k,v in cpu(a.policy.state_dict()).items())
        assert a.step==3 and a.scheduler.last_epoch==3
    finally:dist.destroy_process_group()
    verify();write(O/'GPU_PREFLIGHT.json',dict(status='PASS',isolated_optimizer_calls=4,formal_rollouts=0,environment_steps=0,selection=out,gpu=torch.cuda.get_device_name(),torch=torch.__version__,seconds=time.time()-begin,modal_semantics='Inherited unchanged complete_policy/model/eval_metric with parent PASS; real latent GPU update tested here',contract_sha256=sha(O/'CONTRACT.json')))
if __name__=='__main__':main()
