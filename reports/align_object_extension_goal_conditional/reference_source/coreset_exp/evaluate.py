from .common import SEED,SELECTOR_SEED
import argparse,json,os,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .common import O,write,save,sha,verify,rng
from .model import cfg,data_setup,complete,strict_downstream
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
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['ra_coreset']);args=ap.parse_args()
    verify();assert (O/args.arm/'TRAINING_COMPLETE.json').exists()
    root=O/args.arm/'evaluation';root.mkdir(exist_ok=True)
    assert not list(root.glob('trials*.jsonl')) and not list(root.glob('attempts*.jsonl')),'Partial evaluation requires audited RNG-preserving recovery'
    c=cfg('eval');c.experiment_dir=str(root);control_seed(SEED);b,ds,shape=data_setup(c,True);c.shape_meta=shape
    write(root/'resolved_actual.json',dict(c));torch.backends.cudnn.benchmark=True
    from . import evaluate as context
    from .eval_metric import evaluate_one_task_success
    for stage in range(6,10):
        control_seed(SEED)
        for t in range(6,stage+1):
            checkpoint=O/args.arm/'models'/f'mlrifa_{stage}_model_ep99.pth'
            raw=torch.load(checkpoint,map_location='cpu');assert raw['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
            assert raw['contract_sha256']==sha(O/'CONTRACT.json')
            algo=Basic(10,c).cuda();n=strict_downstream(algo.policy,checkpoint);algo.eval()
            task=b.get_task(t);init_path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file
            init=torch.load(init_path);assert len(init)>=20
            context.CONTEXT.clear();context.CONTEXT.update(arm=args.arm,stage=stage,epoch=99,checkpoint_sha256=sha(checkpoint),
                modal_sha256=sha(O/'MODAL_CONTRACT.json'),loaded_downstream_keys=n,
                initial_file_sha256=sha(init_path),initial_hashes=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in init[:20]],
                trial_log=str(root/f'trials_stage{stage}_task{t}.jsonl'),attempt_log=str(root/f'attempts_stage{stage}_task{t}.jsonl'))
            save(root/f'rng_before_stage{stage}_task{t}.pth',rng())
            try:
                instr,mask=b.get_task_emb(t)
                result=evaluate_one_task_success(c,algo,task,instr,mask,t,see=SEED)
                write(root/f'cell_stage{stage}_task{t}.json',dict(success_rate=float(result),**context.CONTEXT))
            except BaseException as e:
                context.ledger('technical_exception',exception=repr(e));raise
            finally:del algo;torch.cuda.empty_cache()
            write(O/'RUN_STATUS.json',dict(status='EVALUATING',arm=args.arm,stage=stage,task=t,updated_unix=time.time()))
    write(root/'EVALUATION_COMPLETE.json',dict(status='EVALUATED',episodes=200))
if __name__=='__main__':main()
