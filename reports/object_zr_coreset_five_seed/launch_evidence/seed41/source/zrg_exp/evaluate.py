import argparse,json,os,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .common import O,write,save,sha,verify,rng
from .model import cfg,data_setup,complete,load_stage
from .common import P,ROOT,ARM,numeric,assert_parent_numeric,SEED
from libero.lifelong.utils import torch_load_model
from libero.lifelong.algos.basic import Basic
from libero.lifelong.utils import control_seed
CONTEXT={}
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def ledger(event,**fields):append(CONTEXT['attempt_log'],dict(CONTEXT,event=event,unix=time.time(),**fields))
def record_trial(c,task,seed,trial,initial_state,success,steps):
    append(CONTEXT['trial_log'],dict(CONTEXT,task=task,seed=seed,trial=trial,initial_state_index=initial_state,
        initial_state_sha256=CONTEXT['initial_hashes'][initial_state],success=success,vector_group_steps=steps))
def main():
    ap=argparse.ArgumentParser();args=ap.parse_args();args.arm=ARM
    verify();assert (O/args.arm/'TRAINING_COMPLETE.json').exists()
    root=O/args.arm/'evaluation';root.mkdir(exist_ok=True)
    assert not list(root.glob('trials*.jsonl')) and not list(root.glob('attempts*.jsonl')),'Partial evaluation requires audited RNG-preserving recovery'
    c=cfg('eval');c.experiment_dir=str(root);control_seed(SEED);b,ds,shape=data_setup(c,True);c.shape_meta=shape
    write(root/'resolved_actual.json',dict(c));torch.backends.cudnn.benchmark=True;write(O/'NUMERIC_EVAL.json',assert_parent_numeric(True));started=time.time()
    from . import evaluate as context
    from .eval_metric import evaluate_one_task_success
    for stage in range(6,10):
        control_seed(SEED)
        for t in range(6,stage+1):
            checkpoint=O/args.arm/'models'/f'mlrifa_{stage}_model_ep99.pth'
            raw=torch.load(checkpoint,map_location='cpu');assert raw['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
            assert raw['contract_sha256']==sha(O/'CONTRACT.json')
            assert raw['model_contract_sha256']==sha(O/'MODEL_CONTRACT.json')
            algo=Basic(10,c).cuda()
            official=torch_load_model(c.pretrain)[0];actual=algo.policy.state_dict()
            assert set(official)==set(actual) and all(torch.equal(official[k].cpu(),actual[k].cpu()) for k in actual)
            n=load_stage(algo.policy,checkpoint,stage);algo.eval()
            task=b.get_task(t);init_path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file
            init=torch.load(init_path);assert len(init)>=20
            context.CONTEXT.clear();context.CONTEXT.update(arm=args.arm,stage=stage,epoch=99,checkpoint_sha256=sha(checkpoint),
                modal_sha256=sha(O/'MODAL_CONTRACT.json'),model_contract_sha256=sha(O/'MODEL_CONTRACT.json'),loaded_downstream_keys=n,
                initial_file_sha256=sha(init_path),initial_hashes=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in init[:20]],
                trial_log=str(root/f'trials_stage{stage}_task{t}.jsonl'),attempt_log=str(root/f'attempts_stage{stage}_task{t}.jsonl'))
            save(root/f'rng_before_stage{stage}_task{t}.pth',rng())
            try:
                instr,mask=b.get_task_emb(t)
                result=evaluate_one_task_success(c,algo,task,instr,mask,t,see=SEED)
                latency=algo.policy.action_latency
                write(root/f'cell_stage{stage}_task{t}.json',dict(success_rate=float(result),get_action_calls=len(latency),get_action_seconds_sum=sum(latency),get_action_seconds_mean=float(np.mean(latency)),get_action_seconds_p95=float(np.percentile(latency,95)),**context.CONTEXT))
            except BaseException as e:
                context.ledger('technical_exception',exception=repr(e));raise
            finally:
                save(root/f'rng_after_stage{stage}_task{t}.pth',rng());del algo;torch.cuda.empty_cache()
            write(O/'RUN_STATUS.json',dict(status='EVALUATING',arm=args.arm,stage=stage,task=t,updated_unix=time.time()))
    write(root/'EVALUATION_COMPLETE.json',dict(status='EVALUATED',episodes=200,seconds=time.time()-started))
if __name__=='__main__':main()
