"""Original parent rollout function, only a new policy readout and provenance."""
from .model import *
from coreset_exp.model import data_setup
from coreset_exp import evaluate as context
from libero.lifelong.algos.basic import Basic

def main():
    verify();assert (O/'CALIBRATED.json').exists();setup()
    root=O/'evaluation';root.mkdir(exist_ok=True)
    assert not list(root.glob('trials*.jsonl')) and not list(root.glob('attempts*.jsonl')),'Partial evaluation requires audited RNG/environment recovery'
    c=cfg('eval');c.experiment_dir=str(root);control_seed(43);b,ds,shape=data_setup(c,True);c.shape_meta=shape
    write(root/'resolved_actual.json',dict(c));torch.backends.cudnn.benchmark=True
    from coreset_exp.eval_metric import evaluate_one_task_success
    t0=None
    for stage in range(6,10):
        control_seed(43)
        cp=O/'models'/f'zrv2_stage{stage}_calibration_ep9.pth';snap=torch.load(cp,map_location='cpu')
        assert snap['task']==stage and snap['next_epoch']==10 and snap['contract_sha256']==sha(O/'CONTRACT.json')
        parent=snap['parent'];assert sha(parent['path'])==parent['sha256']
        for taskid in range(6,stage+1):
            algo=Basic(10,c).cuda();n=strict_downstream(algo.policy,parent['path']);algo.eval();algo.requires_grad_(False)
            r=cpu(rng())
            if t0 is None:
                temp,t0=new_public();del temp
            bank,_=make_bank(stage);bank.load_state_dict(snap['bank'],strict=True);bank.requires_grad_(False)
            assert tensor_hash(t0)==snap['T0_hash'];ctx=install_native(algo.policy,t0,bank);algo.eval();restore_rng(r)
            task=b.get_task(taskid);init_path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;init=torch.load(init_path)
            context.CONTEXT.clear();context.CONTEXT.update(arm='coreset_ifa_zrv2',stage=stage,parent_epoch=99,calibration_epoch=9,
                parent_checkpoint_sha256=parent['sha256'],checkpoint_sha256=sha(cp),T0_hash=snap['T0_hash'],gate_hash=hashlib.sha256(bank.gate_w.cpu().numpy().tobytes()).hexdigest(),
                modal_sha256=sha(P/'MODAL_CONTRACT.json'),loaded_downstream_keys=n,initial_file_sha256=sha(init_path),initial_hashes=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in init[:20]],
                trial_log=str(root/f'trials_stage{stage}_task{taskid}.jsonl'),attempt_log=str(root/f'attempts_stage{stage}_task{taskid}.jsonl'))
            save(root/f'rng_before_stage{stage}_task{taskid}.pth',rng())
            status('EVALUATING',stage=stage,task=taskid)
            try:
                instr,mask=b.get_task_emb(taskid);tic=time.time();result=evaluate_one_task_success(c,algo,task,instr,mask,taskid,see=43)
                write(root/f'cell_stage{stage}_task{taskid}.json',dict(success_rate=float(result),seconds=time.time()-tic,spatial_calls=ctx['spatial_calls'],**context.CONTEXT))
            except BaseException as exc:context.ledger('technical_exception',exception=repr(exc));raise
            finally:del algo,bank;torch.cuda.empty_cache()
    write(root/'EVALUATION_COMPLETE.json',dict(status='EVALUATED',episodes=200));status('EVALUATED')
if __name__=='__main__':main()
