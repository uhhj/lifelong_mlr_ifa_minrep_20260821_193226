"""Original rollout rule, 75 task cells; five stage RNG resets and no base-task testing."""
import json
import time
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from libero.lifelong.algos.basic import Basic
from libero.lifelong.utils import control_seed,torch_load_model
from .common import O,SEED,verify,write,sha,save,rng
from .data import cfg,lifelong_datasets,tasks,native_task
CONTEXT={}
def append(p,row):
    with Path(p).open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def ledger(event,**fields):append(CONTEXT["attempt_log"],dict(CONTEXT,event=event,unix=time.time(),**fields))
def record_trial(c,task,seed,trial,initial_state,success,steps):
    append(CONTEXT["trial_log"],dict(CONTEXT,task=task,seed=seed,trial=trial,
        initial_state_index=initial_state,initial_state_sha256=CONTEXT["initial_hashes"][initial_state],
        success=success,vector_group_steps=steps))
def main():
    verify();assert (O/"ra_coreset/TRAINING_COMPLETE.json").exists()
    root=O/"ra_coreset/evaluation";root.mkdir(exist_ok=True)
    assert not list(root.glob("trials*.jsonl")) and not list(root.glob("attempts*.jsonl")),"No blind partial global-RNG evaluation restart"
    c=cfg("eval");c.experiment_dir=str(root);control_seed(SEED)
    ds,emb,shape=lifelong_datasets(c);c.shape_meta=shape
    write(root/"resolved_actual.json",dict(c));torch.backends.cudnn.benchmark=True
    from . import evaluate as context
    from .eval_metric import evaluate_one_task_success
    rows=tasks()
    for stage in range(1,6):
        control_seed(SEED)
        for t in range(stage*5):
            checkpoint=O/"ra_coreset/models"/("mlrifa_stage%d_ep99.pth"%stage)
            raw=torch.load(checkpoint,map_location="cpu")
            assert raw["contract_sha256"]==sha(O/"CONTRACT.json")
            assert raw["pretrain_sha256"]==sha(c.pretrain)
            algo=Basic(25,c).cuda()
            source=torch_load_model(checkpoint)[0]
            keys={k for k in algo.policy.state_dict() if k.startswith(("temporal_transformer.","policy_head."))}
            assert set(source)==keys
            loaded=algo.policy.load_state_dict(source,strict=False)
            assert not loaded.unexpected_keys and not (keys&set(loaded.missing_keys))
            assert all(torch.equal(algo.policy.state_dict()[k].cpu(),source[k]) for k in keys)
            algo.eval();task=native_task(rows[t])
            init_path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file
            init=torch.load(init_path);assert len(init)>=20
            context.CONTEXT.clear();context.CONTEXT.update(arm="ra_coreset",stage=stage,epoch=99,
                checkpoint_sha256=sha(checkpoint),loaded_downstream_keys=len(keys),
                pretrain_sha256=sha(c.pretrain),initial_file_sha256=sha(init_path),
                initial_hashes=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in init[:20]],
                trial_log=str(root/("trials_stage%d_task%d.jsonl"%(stage,t))),
                attempt_log=str(root/("attempts_stage%d_task%d.jsonl"%(stage,t))))
            save(root/("rng_before_stage%d_task%d.pth"%(stage,t)),rng())
            try:
                result=evaluate_one_task_success(c,algo,task,emb["input_ids"][t],
                                                 emb["attention_mask"][t],t,see=SEED)
                write(root/("cell_stage%d_task%d.json"%(stage,t)),dict(success_rate=float(result),**context.CONTEXT))
            except BaseException as exc:context.ledger("technical_exception",exception=repr(exc));raise
            finally:del algo;torch.cuda.empty_cache()
            write(O/"RUN_STATUS.json",dict(status="EVALUATING",stage=stage,task=t,unix=time.time()))
    write(root/"EVALUATION_COMPLETE.json",dict(status="EVALUATED",episodes=1500,vector_attempts=150))
if __name__=="__main__":main()
