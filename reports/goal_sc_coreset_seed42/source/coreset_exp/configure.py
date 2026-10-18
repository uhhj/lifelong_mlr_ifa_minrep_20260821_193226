from .common import SEED,SELECTOR_SEED
import json,os,time,inspect,shutil
from pathlib import Path
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from easydict import EasyDict
from .common import S,O,write,sha,source_hashes

def make_config(kind,arm=None):
    train=kind=='train'
    overrides=[f'seed={SEED}','benchmark_name=LIBERO_GOAL','data.task_order_index=0','data.demo_limit=10','data.seq_len=8',
        'policy='+('BriefPolicy' if train else 'complete_policy'),'lifelong='+('mlr' if train else 'basic'),
        'pretrain=/data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth',
        'folder=/data/mlr_ifa/datasets','bddl_folder=/data/mlr_ifa/src/libero/libero/bddl_files',
        'init_states_folder=/data/mlr_ifa/src/libero/libero/init_files',
        '+start_task=6','+end_task=10','+gamma=0.7','+past_data_rate=0.5',
        '+aug='+('false' if train else 'true'),'training='+('true' if train else 'false'),
        '++policy.use_vision=true','++policy.use_language=true','++policy.use_state=true',
        'train.batch_size='+('1' if kind=='export' else '10'),'train.n_epochs=100','train.num_workers=0',
        'eval.n_eval=20','eval.max_steps=600','+feature_dir='+str(O/'features'),
        '+buffer_dir='+str(O/(arm or kind)/'buffer'),
        '+experiment_dir='+str(O/(arm or kind)/'models')]
    with initialize_config_dir(config_dir=str(S/'libero/configs'),version_base=None):
        c=compose(config_name='config_adaption' if kind!='eval' else 'test',overrides=overrides)
    cfg=OmegaConf.to_container(c,resolve=True)
    cfg['shape_meta']=None;cfg['selector']=arm or 'none';cfg['run_root']=str(O/(arm or kind))
    cfg['contract']=str(O/'CONTRACT.json');cfg['linear_schedule_unit']='optimizer_step'
    for k in ('use_vision','use_language','use_state'):assert cfg['policy'][k] is True
    return cfg,overrides

def main():
    import torch,robomimic.utils.dataset
    from libero.libero.benchmark import get_benchmark
    assert torch.cuda.is_available();O.mkdir(parents=True,exist_ok=True)
    assert not (O/'CONTRACT.json').exists(),'Do not overwrite frozen contract'
    b=get_benchmark('LIBERO_GOAL')(0);inputs={};tasks={}
    for t in range(10):
        task=b.get_task(t);paths=[Path('/data/mlr_ifa/datasets')/b.get_task_demonstration(t),
            Path('/data/mlr_ifa/src/libero/libero/bddl_files')/task.problem_folder/task.bddl_file,
            Path('/data/mlr_ifa/src/libero/libero/init_files')/task.problem_folder/task.init_states_file]
        for p in paths:assert p.is_file(),str(p);inputs[str(p)]=sha(p)
        tasks[str(t)]=dict(name=task.name,paths=list(map(str,paths)))
    pt=Path('/data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth');inputs[str(pt)]=sha(pt)
    assert inputs[str(pt)]=='241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7'
    parser=Path(inspect.getfile(robomimic.utils.dataset));inputs[str(parser)]=sha(parser)
    configs={};commands={}
    for kind,arm in [('export',None),('train','random'),('train','coreset'),('eval',None)]:
        c,cmd=make_config(kind,arm);name=arm or kind;p=O/f'resolved_{name}.json';write(p,c);configs[str(p)]=sha(p);commands[name]=cmd
    checkpoint=torch.load(pt,map_location='cpu');pcfg=checkpoint.get('cfg',{});pp=pcfg.get('policy',{})
    write(O/'PRETRAIN_METADATA.json',dict(pretrain_policy={k:str(v) for k,v in pp.items()},
        pretrain_use_state=pp.get('use_state','unknown'),effective_use_state=True,
        gap='DISCLOSED_GAP' if 'use_state' not in pp else None))
    write(O/'CONFIG_COMMANDS.json',commands);write(O/'TASKS.json',tasks)
    modal=dict(flags=dict(vision=True,language=True,state=True),token_order=['language','agentview','eye_in_hand','state'],
        state_fields=['joint_states','gripper_states'],state_dims=[7,2],window=8,embedding=512,
        frontend_sha256=inputs[str(pt)],base_commit='a721807f2f7fa7f9833dcc85c592d78e9accd00b',
        source_sha256=source_hashes(),export_preprocessing='native BasePolicy augmentation, Basic.learn_one_task train mode, batch1 RandomSampler',
        evaluation_preprocessing='native CompletePolicy.get_action with eval mode (augment group identity)')
    write(O/'MODAL_CONTRACT.json',modal)
    protected={}
    for root in ('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run', *[f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013' for s in (42,43,44)]):
        for p in Path(root).rglob('*'):
            if p.is_file() and O not in p.parents and p.suffix in ('.json','.jsonl','.csv','.pth'):
                protected[str(p)]=sha(p)
    write(O/'OLD_INPUTS_READONLY.json',protected)
    write(O/'CONTRACT.json',dict(protocol='paper_aligned_GOAL_three_seed_v1',seed=SEED,arms=['random','coreset'],
        tasks=[6,7,8,9],epochs_each=100,batch=10,gamma=.7,ifa_weight=.1,formal_rollouts=400,
        source_sha256=source_hashes(),inputs=inputs,configs=configs,modal_sha256=sha(O/'MODAL_CONTRACT.json'),
        old_readonly_sha256=sha(O/'OLD_INPUTS_READONLY.json'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        extensions=[42,43,44],selector_seed=SELECTOR_SEED,extension_authorization='User authorized GOAL Random/Coreset seeds42/43/44; system disk outputs; fresh paired baselines due disclosed historical differences',warmup_provenance='500 release fallback; scheduler each optimizer step is disclosed engineering interpretation',created_unix=time.time(),suite='LIBERO_GOAL',expected_windows=[997,889,904,1969],expected_steps_per_arm=76100,orchestration_sha256=sha(S.parents[1]/'ORCHESTRATION_SHA256.json')))
    write(O/'RUN_STATUS.json',dict(status='PREPARED',updated_unix=time.time()))
if __name__=='__main__':main()
