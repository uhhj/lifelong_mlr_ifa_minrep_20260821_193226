"""Native frozen-feature export with identities from actual accessed dataset indices."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,RandomSampler
import robomimic.utils.tensor_utils as TensorUtils
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.utils import control_seed
from .common import O,SEED,verify,sha,write,save,rng,restore_rng
from .data import cfg,lifelong_datasets,Indexed

def identity(dataset,index):
    d=dataset.sequence_dataset
    demo=d._index_to_demo_id[index]
    start=index-d._demo_id_to_start_indices[demo]
    length=d._demo_id_to_demo_length[demo]
    assert 0<=start<length
    return dict(dataset_index=index,demo_id=demo,start=start,target=min(start+7,length-1),length=length)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    verify();assert json.loads((O/"GPU_PREFLIGHT.json").read_text())["status"]=="PASS"
    assert not (O/"FEATURES.json").exists()
    c=cfg("export");control_seed(SEED);datasets,emb,shape=lifelong_datasets(c);c.shape_meta=shape
    model=CompletePolicy(c,shape).cuda();model.train()
    cp=O/"export_recovery.pth";start=0;records={}
    if args.resume:
        old=torch.load(cp,map_location="cpu")
        assert old["contract_sha256"]==sha(O/"CONTRACT.json")
        start=old["next_task"];records=old["records"];restore_rng(old["rng"])
    else:assert not list((O/"features").glob("*/*.pt"))
    began=time.time()
    for task in range(start,25):
        target=O/"features"/str(task);target.mkdir(parents=True,exist_ok=True)
        assert not list(target.glob("*.pt")),"Partial task export requires explicit preserved-attempt recovery"
        dataset=datasets[task]
        loader=DataLoader(Indexed(dataset),batch_size=1,sampler=RandomSampler(dataset),
                          num_workers=c.train.num_workers,persistent_workers=False)
        bindings={};hashes={}
        with torch.no_grad():
            for i,(data,indices) in enumerate(loader):
                index=int(indices[0]);meta=identity(dataset,index)
                data=TensorUtils.map_tensor(data,lambda x:x.cuda(non_blocking=True))
                data=model.preprocess_input(data)
                feature=model.spatial_encode(data)
                assert feature.shape==(1,8,4,512) and torch.isfinite(feature).all()
                name=str(i)+".pt";path=target/name
                save(path,dict(feature=feature.cpu(),actions=data["actions"].cpu()))
                hashes[name]=sha(path);bindings[name]=dict(meta,sha256=hashes[name])
        assert len(bindings)==len(dataset) and len({x["dataset_index"] for x in bindings.values()})==len(dataset)
        write(O/"bindings"/("task%d.json"%task),dict(task=task,windows=bindings,
            source="actual native DataLoader index, no second sampler",
            demos=list(dataset.sequence_dataset.demos)))
        records[str(task)]=dict(count=len(dataset),hashes=hashes,demos=list(dataset.sequence_dataset.demos))
        save(cp,dict(next_task=task+1,records=records,rng=rng(),contract_sha256=sha(O/"CONTRACT.json")))
        write(O/"RUN_STATUS.json",dict(status="EXPORTING",task=task,unix=time.time(),pid=os.getpid()))
    write(O/"FEATURES.json",dict(tasks=records,export_mode="original train mode/augmentation/RandomSampler batch1",
        seed=SEED,seconds_this_invocation=time.time()-began,checkpoint_sha256=sha(c.pretrain)))
if __name__=="__main__":main()
