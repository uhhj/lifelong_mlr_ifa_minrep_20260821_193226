"""Read-only targeted asset audit and trajectory-role registration."""
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from .common import ROOT, OUT, STORE, TASKS, SEEDS, sha, write, freeze, split_demos, balanced_windows


def main():
    from cb_ifa_experiment import gpu_runtime as env
    import h5py
    import torch
    from libero.libero.benchmark import get_benchmark
    from .statistics import folds
    OUT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    cfg = env.train_config(45, STORE / 'configuration_only')
    bench = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    base = Path(cfg.pretrain)
    context = dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        base_checkpoint=str(base), base_sha256=sha(base), diagnostic_storage=str(STORE),
        diagnostic_training_performed=False, formal_training_resumed=False)
    write(OUT / 'EXPERIMENT_CONTEXT.json', context)
    tasks = {}
    for task in TASKS:
        path = ROOT / 'data' / bench.get_task_demonstration(task)
        with h5py.File(path, 'r') as f:
            demos = sorted(f['data'], key=lambda d: int(d.split('_')[-1]))
            lengths = {d: len(f['data'][d]['actions']) for d in demos}
            fingerprints = {d: hashlib.sha256(f['data'][d]['actions'][:].tobytes()).hexdigest() for d in demos}
        roles = split_demos(task, demos)
        offsets = {}; offset = 0
        for d in demos:
            offsets[d] = offset; offset += lengths[d]
        windows = {}
        for role in roles:
            pairs = ([(d, i) for d in roles[role] for i in range(lengths[d])]
                     if role == 'A_train' else balanced_windows(roles[role], lengths, 128 if role == 'D_signal' else 320))
            windows[role] = [dict(task=task, trajectory_id=d, window_index=i,
                                  source_record_index=offsets[d]+i, trajectory_length=lengths[d],
                                  valid_frame_count=min(8, lengths[d]-i)) for d, i in pairs]
        tasks[str(task)] = dict(language=bench.get_task(task).language, path=str(path),
                               file_size=path.stat().st_size, demos=roles, lengths=lengths,
                               action_sha256=fingerprints, windows=windows)
    freeze(OUT / 'DATA_MANIFEST.json', dict(tasks=tasks, split='SHA256 task_relation_forecast_v1/split/task/demo; independent of optimization seed',
           signal_access=['source D_signal', 'target D_signal', 'target first A_train batch for shadow only'],
           evaluator_access=['E_eval'], window_definition='Native SequenceDataset forward window, pad end; valid-frame mask excludes padded suffix'))
    rows = [dict(seed=seed, source=s, target=t, pilot=seed==45 and (s,t) in ((6,7),(7,6)))
            for seed in SEEDS for s in TASKS for t in TASKS if s != t]
    freeze(OUT / 'PAIR_MANIFEST.json', dict(pairs=rows, units=36))
    freeze(OUT / 'CV_SPLITS.json', dict(strict_leave_target=folds(rows), leave_seed=folds(rows, 'seed')))
    # Same native 100-epoch horizon for every source/target; task6 A_train is
    # the predeclared common reference length, not a shortened 300-step horizon.
    horizon = cfg.train.n_epochs * ((len(tasks['6']['windows']['A_train']) + cfg.train.batch_size-1)//cfg.train.batch_size)
    warmup = cfg.train.scheduler.kwargs.warmup_steps
    lr = cfg.train.optimizer.kwargs.lr
    rates = [lr * (float(k)/max(1,warmup) if k < warmup else max(0.,float(horizon-k)/max(1,horizon-warmup))) for k in range(1001)]
    freeze(OUT / 'TRAINING_PROTOCOL.json', dict(optimizer=cfg.train.optimizer, batch_size=cfg.train.batch_size,
           grad_clip=cfg.train.grad_clip, loss_scale=cfg.train.loss_scale, common_schedule_horizon=horizon,
           horizon_source='native100 epochs times ceil(task6 A_train windows / native batch10), common to all tasks',
           warmup_steps=warmup, step_lr=rates, lr_sha256=hashlib.sha256(json.dumps(rates).encode()).hexdigest(),
           native_scheduler='batch_step linear warmup and linear decay, no short-budget compression',
           fresh_optimizer=True, replay=False, extra_losses=False, frontend_frozen=True,
           dropout_stream='stable_seed(protocol,train,seed,target,step), identical across source branches',
           augmentation='native preprocess_input during A_train feature extraction, paired cache per seed/task',
           D_E_evaluation='deterministic frontend eval; no stochastic augmentation'))
    pretraining = ROOT / 'libero/lifelong/pretraining.py'
    text = pretraining.read_text()
    assert 'n_manip_tasks = 6' in text and 'range(n_manip_tasks)' in text
    audit = dict(status='RELEASE_RECIPE_AND_CHECKPOINT_PROVENANCE_CONSISTENT',
                 robot_bc_tasks=list(range(6)), held_tasks=list(TASKS),
                 evidence=[dict(path=str(pretraining), sha256=sha(pretraining), finding='n_manip_tasks=6 and dataset loop range(n_manip_tasks), task_order_index0'),
                           dict(path=str(ROOT/'REPRODUCTION.md'), sha256=sha(ROOT/'REPRODUCTION.md'), finding='Official author-provided unmodified OBJECT pretraining checkpoint; six pretrained tasks')],
                 actual_pretraining_job_log_available=False,
                 limitation='Archived official checkpoint and released recipe agree; original author training-job manifest is not available. This is provenance-based exposure verification, not direct access to the original training log.',
                 shared_official_pretraining_seed=42, independent_pretraining_seeds=False,
                 candidate_tasks_exposed_in_released_bc_recipe=False)
    write(OUT / 'BASE_EXPOSURE_AUDIT.json', audit)
    provenance = {}
    for name in ['cb_ifa_experiment/gpu_runtime.py', 'gradient_relation_experiment/kernel.py',
                 'gn_functional_geometry_v1/engine.py', 'libero/lifelong/algos/base.py',
                 'libero/lifelong/models/brief.py', 'libero/lifelong/models/complete_policy.py']:
        p = ROOT/name
        provenance[name] = dict(path=str(p), sha256=sha(p), git_tracked=subprocess.call(['git','ls-files','--error-unmatch',name],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)==0)
    write(OUT/'RUNTIME_PROVENANCE.json', provenance)
    print(json.dumps(dict(stage='ASSET_AND_SPLIT_PREPARATION_COMPLETE', tasks={t:{r:len(v) for r,v in x['windows'].items()} for t,x in tasks.items()}, free_bytes=shutil.disk_usage(STORE).free, formal_training_resumed=False), indent=2))


if __name__ == '__main__':
    main()
