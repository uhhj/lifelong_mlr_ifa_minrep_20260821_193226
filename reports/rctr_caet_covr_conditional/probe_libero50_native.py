"""CPU-only native dataset, constructor and temporal/GMM interface admission."""
import json
import io
import os
from pathlib import Path
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from libero.lifelong.datasets import get_dataset,SequenceVLDataset
from libero.lifelong.utils import get_task_embs,control_seed
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.models.brief import BriefPolicy
from l50_exp.data import native_task
from l50_exp.export import identity
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
partition=json.loads((Q/"LIBERO50_PARTITION.json").read_text())
data=Path("/data/mlr_ifa/datasets/libero50_acquired")
c=EasyDict(json.loads((Q/"LIBERO50_PARENT_EXPORT_TEMPLATE.json").read_text()))
c.pretrain=None;c.pretraining_mode=True;c.seed=42;c.gamma=.1;c.policy.use_state=True
c.feature_dir="/unused";c.buffer_dir="/unused";c.experiment_dir="/unused";c.train.num_workers=0
control_seed(42);torch.set_num_threads(4)
rows=partition["pretraining"]+sum(partition["lifelong_stages"],[])
checked=[];first=None
for row in rows:
    path=data/row["dataset_relative_path"]
    if not path.is_file():continue
    d,shape=get_dataset(str(path),c.data.obs.modality,seq_len=8,demo_limit=10,train=True)
    assert d.n_demos==10
    emb=get_task_embs(c,[native_task(row).language])
    vl=SequenceVLDataset(d,emb["input_ids"][0],emb["attention_mask"][0])
    for index in (0,len(d)-1):
        meta=identity(vl,index)
        item=vl[index]
        assert item["actions"].shape==(8,7)
        assert meta["target"]==min(meta["start"]+7,meta["length"]-1)
    checked.append(dict(task=row["name"],windows=len(d),demos=d.n_demos))
    if first is None:first=(vl,shape)
    if len(checked)==3:break
assert len(checked)==3
c.shape_meta=first[1]
model=CompletePolicy(c,c.shape_meta).cpu().eval()
names=[n for n,p in model.named_parameters() if p.requires_grad]
assert any("state_encoder." in n for n in names)
assert any("image_encoders." in n and ".lora_" in n for n in names)
assert any("language_encoder." in n and ".lora_" in n for n in names)
sample=next(iter(DataLoader(first[0],batch_size=1,num_workers=0)))
with torch.no_grad():
    native=model.preprocess_input(sample,train_mode=False)
    x=model.spatial_encode(native)
    assert x.shape==(1,8,4,512) and torch.isfinite(x).all()
    # Native BriefPolicy always loads a parent. Supply this CPU-only initial
    # state in memory; do not write or masquerade it as a pretrained artifact.
    temporary=io.BytesIO()
    torch.save(dict(state_dict=model.state_dict(),cfg=c),temporary)
    temporary.seek(0);c.pretrain=temporary
    brief=BriefPolicy(c,c.shape_meta).cpu().eval()
    subset={k:v for k,v in model.state_dict().items() if k in brief.state_dict()}
    brief.load_state_dict(subset,strict=True)
    for h in (1,4,8):
        a=model.policy_head.forward_fn(model.temporal_encode(x[:,:h])[:,-1])
        b=brief.policy_head.forward_fn(brief.temporal_encode(x[:,:h])[:,-1])
        assert all(torch.equal(u,v) for u,v in zip(a,b))
report=dict(status="CPU_NATIVE_PASS_NOT_GPU_ADMITTED",checked=checked,
    trainable_names=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    native_feature_shape=list(x.shape),history_lengths=[1,4,8],strict_brief_keys=len(subset),
    optimizer_calls=0,environment_steps=0,cuda_allocated_bytes=torch.cuda.memory_allocated())
(Q/"LIBERO50_CPU_NATIVE.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="trainable_names"}))
