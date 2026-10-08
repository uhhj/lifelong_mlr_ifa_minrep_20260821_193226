"""Freeze the existing paired baseline and inputs, without rerunning it."""
import dataclasses
import gzip
import json
import os
import platform
import subprocess
from pathlib import Path

from .control import Settings
from .io import sha, write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e1_object_v1'


def config(seed, run_root=None):
    p=ROOT/('runs/final_paper_text_faithful/ifa/seed%d/run_001/config.json'%seed)
    cfg=json.loads(p.read_text())
    run=Path(run_root) if run_root else ROOT/('runs/e1_object_v1/seed%d'%seed)
    cfg.update(seed=seed,pretrain=str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        feature_dir=str(ROOT/('features/object_seed%d'%seed)),buffer_dir=str(run/'buffer'),
        experiment_dir=str(run/'run_001'),experiment_root=str(run),folder=str(ROOT/'data'),
        bddl_folder=str(ROOT/'libero/libero/bddl_files'),init_states_folder=str(ROOT/'libero/libero/init_files'),
        e1_root=str(run),e1_report=str(OUT/('seed%d'%seed)),e1_recovery=str(run/'recovery/latest.pt'),
        e1_protocol=str(OUT/'PROTOCOL.json'),use_wandb=False,repro_arm='E1_OBJECT_v1',training=True)
    return cfg


def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        print('Existing protocol retained');return
    baselines={str(seed):json.loads((ROOT/('reports/final_paper_text_faithful/primary_ep99/ifa_seed%d.json'%seed)).read_text()) for seed in (42,43,44)}
    checkpoint=ROOT/'checkpoints/object_multitask_model_ep99.pth'
    migration={}
    with gzip.open(ROOT/'.migration/CORE_MANIFEST.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['path'].startswith('features/object_seed') or r['path']=='checkpoints/object_multitask_model_ep99.pth':migration[r['path']]=r['sha256']
    checkpoint_hash=sha(checkpoint)
    assert checkpoint_hash==migration['checkpoints/object_multitask_model_ep99.pth']
    manifest={}
    for seed in (42,43,44):
        manifest[str(seed)]={}
        for task in range(6,10):
            ps=sorted((ROOT/('features/object_seed%d/%d'%(seed,task))).glob('*.pt'))
            assert ps
            manifest[str(seed)][str(task)]=[dict(path=str(p),size=p.stat().st_size,sha256=sha(p)) for p in ps]
            for row in manifest[str(seed)][str(task)]:
                relative=str(Path(row['path']).relative_to(ROOT))
                if relative in migration: assert row['sha256']==migration[relative],relative
                row['historical_migration_hash_match']=relative in migration
    write(OUT/'FEATURE_MANIFEST.json',manifest)
    baseline_protocol=ROOT/'reports/final_paper_text_faithful/PROTOCOL.json'
    protocol=dict(name='E1-OBJECT-v1',seed_order=[42,43,44],tasks=[6,7,8,9],settings=dataclasses.asdict(Settings()),
        baseline_results=baselines,baseline_protocol=json.loads(baseline_protocol.read_text()),
        primary_checkpoint=99,secondary_evaluation='same historical selected-checkpoint protocol; diagnostic only',
        append_seeds_if='seed42 primary overall AUC strictly exceeds seed42 historical IFA primary overall AUC',
        baseline_rerun=False,pretraining_rerun=False,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        feature_manifest_sha256=sha(OUT/'FEATURE_MANIFEST.json'),initial_buffer='empty; task6 has no E1 checks',
        check_sampling='uniform without replacement over sorted legal old buffer windows; matches uniform old-window replay marginal',
        check_rng='random.Random(1000003+seed), independent, reset at each task and saved at epoch boundaries',
        scheduler='inherit paired baseline batch_step, not upstream legacy_epoch',
        reference_auc=baselines['42']['auc'],reference_comparison='same seed; primary ep99; fixed shared pretraining',
        implementation_files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'e1_object_v1').iterdir()) if p.suffix in ('.py','.sh')},
        future_tuning=False)
    protocol['feature_provenance']={'same_paths_as_historical_IFA_baseline':True,
        'matched_migration_hashes':sum(row['historical_migration_hash_match'] for tasks in manifest.values() for rows in tasks.values() for row in rows),
        'total_features':sum(len(rows) for tasks in manifest.values() for rows in tasks.values()),
        'no_feature_regeneration':True,'demonstration_selection':'inherit original cached windows; no re-sampling of demonstrations'}
    write(OUT/'PROTOCOL.json',protocol)
    modules=['libero/lifelong/algos/mlr.py','libero/lifelong/algos/base.py','libero/lifelong/models/brief.py',
        'libero/lifelong/models/policy_head.py','libero/lifelong/models/base_policy.py',
        'libero/lifelong/models/modules/transformer_modules.py','libero/lifelong/test.py','libero/lifelong/metric.py']
    write(OUT/'EXPERIMENT_CONTEXT.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
        python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        files={f:sha(ROOT/f) for f in modules},baseline_protocol_sha256=sha(baseline_protocol),
        baseline_reused=True,old_experiment_queues_resumed=False,training_authorization='E1 only; seed42 then conditional43/44'))
    for seed in (42,43,44): write(OUT/('CONFIG_seed%d.json'%seed),config(seed))
    print('PROTOCOL_FROZEN',sha(OUT/'PROTOCOL.json'))


if __name__=='__main__':main()
