"""Freeze new LIBERO50 contracts only after the exact acquired data are available."""
import argparse
import copy
import json
import os
from pathlib import Path
from easydict import EasyDict
from libero.lifelong.datasets import get_dataset
from .common import W,O,S,SEED,sha,write

Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
DATA=Path("/data/mlr_ifa/datasets/libero50_acquired")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pretrain",action="store_true");a=ap.parse_args()
    assert not (O/"CONTRACT.json").exists(),"Do not reconfigure a frozen run"
    assert not O.exists() or not any(O.iterdir())
    acquisition=json.loads((Q/"LIBERO50_SERVER_STATUS.json").read_text())
    assert acquisition["status"]=="DATA_VERIFIED" and acquisition["verified"]==50
    partition=json.loads((W/"PARTITION.json").read_text())
    O.mkdir(parents=True,exist_ok=True)
    files={str(W/"PARTITION.json"):sha(W/"PARTITION.json")}
    allrows=partition["pretraining"]+sum(partition["lifelong_stages"],[])
    for row in allrows:
        record=Q/"libero50_acquisition"/(Path(row["dataset_relative_path"]).stem+".json")
        proof=json.loads(record.read_text());path=DATA/row["dataset_relative_path"]
        assert proof["upstream_sha256"]==row["raw_sha256"]
        assert proof["all_dataset_bytes_equal"] and proof["all_attributes_equal"]
        assert sha(path)==proof["packed_sha256"]
        files[str(path)]=proof["packed_sha256"];files[str(record)]=sha(record)
    clip=Path(os.environ["HF_HOME"])/"hub/models--openai--clip-vit-base-patch16"
    revision=(clip/"refs/main").read_text().strip()
    files[str(clip/"refs/main")]=sha(clip/"refs/main")
    snapshot=clip/"snapshots"/revision
    assert snapshot.is_dir()
    for p in snapshot.iterdir():
        if p.is_file():files[str(p)]=sha(p)
    pretrain_file=None
    if not a.pretrain:
        shared=Path(os.environ["LIBERO50_PRETRAIN_OUT"])
        accepted=json.loads((shared/"PRETRAIN_STRICT_LOAD.json").read_text())
        assert accepted["status"]=="PASS"
        pretrain_file=accepted["checkpoint"]
        assert sha(pretrain_file)==accepted["sha256"]
        files[str(shared/"PRETRAIN_STRICT_LOAD.json")]=sha(shared/"PRETRAIN_STRICT_LOAD.json")
        files[pretrain_file]=accepted["sha256"]
    templates={"pretrain":"EXPORT"} if a.pretrain else {"export":"EXPORT","train":"TRAIN","eval":"EVAL"}
    configs={}
    for kind,template in templates.items():
        template_path=W/("PARENT_"+template+"_TEMPLATE.json")
        c=EasyDict(json.loads(template_path.read_text()))
        files[str(template_path)]=sha(template_path)
        c.update(seed=42 if a.pretrain else SEED,benchmark_name="LIBERO50_KITCHEN_FIXED",
            folder=str(DATA),gamma=.1,pretrain=pretrain_file,pretraining_mode=a.pretrain,
            feature_dir=str(O/"features"),buffer_dir=str(O/"ra_coreset/buffer"),
            run_root=str(O/"ra_coreset"),experiment_dir=str(O/("pretraining" if a.pretrain else "ra_coreset/models")),
            contract=str(O/"CONTRACT.json"),start_task=0,end_task=25)
        c.policy.use_state=True;c.data.task_group_size=25 if a.pretrain else 5
        c.policy.image_encoder.network_kwargs.vision_model_name=str(snapshot)
        c.policy.language_encoder.network_kwargs.text_model_name=str(snapshot)
        c.train.n_epochs=100;c.n_epochs=100;c.train.batch_size=10 if a.pretrain or kind=="train" else (1 if kind=="export" else c.train.batch_size)
        c.train.num_workers=0
        first=partition["pretraining"][0] if a.pretrain else partition["lifelong_stages"][0][0]
        d,shape=get_dataset(str(DATA/first["dataset_relative_path"]),c.data.obs.modality,
            seq_len=8,demo_limit=50 if a.pretrain else 10,train=c.training)
        c.shape_meta=shape
        configs[kind]=dict(c)
        path=O/("resolved_"+kind+".json");write(path,dict(c));files[str(path)]=sha(path)
    for row in allrows:
        template=next(iter(configs.values()))
        for field,rel in (("bddl_folder","bddl_relative_path"),("init_states_folder","init_relative_path")):
            path=Path(template[field])/row[rel];assert path.is_file();files[str(path)]=sha(path)
    source={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob("*"))
        if p.is_file() and p.suffix in (".py",".yaml",".yml",".sh") and "__pycache__" not in p.parts}
    write(O/"CONTRACT.json",dict(version="RA_LIBERO50_reproduced_pretrain_v1",seed=SEED,
        pretraining=a.pretrain,files=files,source_hashes=source,
        shared_pretrain_seed=42,pretrain_checkpoint_origin="new reproduction; not released authors weights",
        downstream=dict(stages=5,new_tasks_each=5,epochs_per_stage=100,episodes=1500,vector_attempts=150),
        immutable_buffer="hardlinks to own new feature files, byte-identical and never modified"))
    write(O/"RUN_STATUS.json",dict(status="CONFIGURED_AWAITING_GPU",pretraining=a.pretrain))
if __name__=="__main__":main()
