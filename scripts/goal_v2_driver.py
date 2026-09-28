"""Execute the registered state-enabled GOAL seed42 pipeline on an existing GPU."""
import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from easydict import EasyDict
from libero.lifelong.runbook_core import atomic_json, sha256
from libero.lifelong.modal_contract import build_contract, validate_flags, validate_contract, check_metadata

MODALITY_OVERRIDES = ['++policy.use_vision=true', '++policy.use_language=true', '++policy.use_state=true']
LABEL = 'goal_seed42_mlr_ifa_paper_aligned_state_v2'


def overrides(args, phase):
    return ['seed=42', 'benchmark_name=LIBERO_GOAL', 'data.task_order_index=0', 'data.demo_limit=10',
        'data.seq_len=8', 'train.n_epochs=100', 'eval.n_eval=20', 'eval.max_steps=600',
        'train.batch_size=' + ('1' if phase=='export' else '10'),
        'train.num_workers=' + ('0' if phase=='adaptation' else '12'),
        'policy=' + ('BriefPolicy' if phase=='adaptation' else 'complete_policy'),
        'lifelong=' + ('mlr' if phase=='adaptation' else 'basic'),
        'training=' + ('true' if phase=='adaptation' else 'false'),
        '++aug=' + ('false' if phase=='adaptation' else 'true'),
        '++gamma=0.7', '++past_data_rate=0.5', '++start_task=6', '++end_task=10',
        '++runbook_v2=true', '++runbook_resume=' + ('true' if args.mode=='resume' else 'false'),
        '++run_id='+LABEL, '++run_dir='+str(args.run_dir),
        'pretrain='+args.pretrained, 'folder='+args.data_root,
        'bddl_folder='+args.bddl_root, 'init_states_folder='+args.init_states_root,
        'hydra.run.dir='+str(args.run_dir), 'hydra.output_subdir=null'] + MODALITY_OVERRIDES


def cfg_for(args, phase):
    with initialize_config_dir(config_dir=str(ROOT/'libero/configs'), version_base=None):
        cfg = compose(config_name=('test' if phase == 'eval' else 'config_adaption'), overrides=overrides(args, phase))
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    specification=OmegaConf.load(ROOT/'configs/experiments/goal_seed42_paper_aligned_state_v2.yaml')
    assert specification.label == LABEL and cfg.seed == specification.seed
    assert cfg.gamma == specification.gamma and cfg.train.n_epochs == specification.n_epochs
    assert cfg.eval.n_eval == specification.n_eval and cfg.policy.use_state is specification.policy.use_state
    validate_flags(cfg)
    return cfg


def command(args, phase):
    entry = {'export': 'save_feature.py', 'adaptation': 'lifelong_adaptation.py', 'eval': 'test.py'}[phase]
    cmd = [sys.executable]
    if phase != 'eval':
        cmd += ['-m','torch.distributed.run','--standalone','--nproc_per_node=1']
    cmd += [str(ROOT/'libero/lifelong'/entry)] + overrides(args, phase)
    return cmd


def run_command(args, cmd, name):
    logdir = args.run_dir/'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir/'commands.log').open('a') as journal:
        journal.write(time.strftime('%Y-%m-%dT%H:%M:%S%z')+' START '+shlex.join(cmd)+'\n')
        journal.flush()
        with (logdir/(name+'.log')).open('a') as log:
            result = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        journal.write(time.strftime('%Y-%m-%dT%H:%M:%S%z')+' END '+name+' EXIT '+str(result.returncode)+'\n')
    if result.returncode:
        from libero.lifelong.runbook_v2 import status
        state = 'TRAINING_COMPLETED_EVAL_PENDING' if name == 'evaluation' else 'FAILED'
        status(args.run_dir, state, failed_command=name, exit_code=result.returncode)
        raise RuntimeError('%s exit=%d; see %s' % (name,result.returncode,logdir/(name+'.log')))


def prepare(args):
    args.run_dir.mkdir(parents=True,exist_ok=True)
    patch = ROOT.parent/'evidence/source.patch'
    existing_patch=args.run_dir/'source.patch'
    if existing_patch.exists() and sha256(existing_patch) != sha256(patch):
        raise RuntimeError('Existing source snapshot differs; preserve it and use a new run directory')
    cfg = cfg_for(args,'export')
    expected = build_contract(cfg,patch)
    target = args.run_dir/'modal_contract.json'
    if target.exists():
        validate_contract(json.loads(target.read_text()),expected)
    else:
        atomic_json(target,expected)
    if not existing_patch.exists():
        shutil.copy2(patch,existing_patch)
    for phase in ['export','adaptation','eval']:
        cfg = cfg_for(args,phase)
        path=args.run_dir/('resolved_'+phase+'.yaml')
        if not path.exists():
            path.write_text(OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(cfg)))))
    if not (args.run_dir/'resolved_config.yaml').exists():
        shutil.copy2(args.run_dir/'resolved_adaptation.yaml',args.run_dir/'resolved_config.yaml')
    return expected


def preflight(args):
    import verify_goal_seed42 as legacy
    cfg = cfg_for(args,'export')
    legacy.preflight(args,cfg)
    manifest = json.loads((args.run_dir/'pretrained_manifest.json').read_text())
    manifest.update(pretrain_use_state=manifest['checkpoint_cfg']['policy'].get('use_state','unknown'),
                    effective_use_state=True, modality_gap_status='DISCLOSED_GAP')
    atomic_json(args.run_dir/'pretrained_manifest.json',manifest)
    # Supersede old preflight status names, not historical audit evidence.
    atomic_json(args.run_dir/'RUN_STATUS.json',{'status':'PREFLIGHT_PASSED','training_epochs_completed':0,
        'formal_episodes_completed':0,'run_id':LABEL,'effective_use_state':True,'pretrain_use_state':'unknown'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','smoke','full','resume','eval'],required=True)
    for name in ['pretrained','data-root','bddl-root','init-states-root']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--smoke-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    args.run_dir=args.run_dir.resolve()
    if args.smoke_dir is None:
        args.smoke_dir=args.run_dir.parent/'smoke_v2'
    os.environ.update(PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
        CUDA_VISIBLE_DEVICES='0',MUJOCO_EGL_DEVICE_ID='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    if args.dry_run:
        for phase in ['export','adaptation','eval']:
            cfg_for(args,phase)
            print(shlex.join(command(args,phase)))
        return
    contract=prepare(args)
    if args.mode=='preflight':
        preflight(args)
    elif args.mode=='smoke':
        assert args.run_dir == args.smoke_dir, 'Smoke must use the isolated smoke directory'
        run_command(args,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
                         str(ROOT/'scripts/smoke_state_v2.py'),'--run-dir',str(args.run_dir)],'smoke')
    else:
        checked=json.loads((args.smoke_dir/'test_gate.json').read_text())
        check_metadata(checked,contract)
        assert checked['status']=='PASS'
        shutil.copy2(args.smoke_dir/'modality_tests.json',args.run_dir/'modality_tests.json')
        if args.mode=='full':
            if (args.run_dir/'features').exists() or (args.run_dir/'checkpoints').exists():
                raise RuntimeError('Full requires fresh features and checkpoints; use a new directory or checked resume')
            if not (args.run_dir/'preflight_result.json').exists():
                preflight(args)
            run_command(args,command(args,'export'),'export')
            run_command(args,command(args,'adaptation'),'adaptation')
        elif args.mode=='resume':
            run_command(args,command(args,'adaptation'),'adaptation_resume')
        run_command(args,command(args,'eval'),'evaluation')
        run_command(args,[sys.executable,str(ROOT/'scripts/summarize_goal_seed42.py'),
                          '--run-dir',str(args.run_dir)],'summarize')
    print('DRIVER_MODE_FINISHED',args.mode,flush=True)


if __name__=='__main__':
    main()
