"""Freeze only after engineering checks; never overwrite previous protocols."""
import json
import platform
import subprocess
from pathlib import Path
from .io import sha,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v1_object_tsr'


def config(seed,run_root=None):
    assert seed==42
    cfg=json.loads((ROOT/('runs/final_paper_text_faithful/ifa/seed%d/run_001/config.json'%seed)).read_text())
    run=Path(run_root) if run_root else ROOT/('runs/v1_object_tsr/seed%d'%seed)
    cfg.update(seed=seed,pretrain=str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        feature_dir=str(ROOT/('features/object_seed%d'%seed)),buffer_dir=str(run/'buffer'),
        experiment_dir=str(run/'checkpoints'),experiment_root=str(run),folder=str(ROOT/'data'),
        bddl_folder=str(ROOT/'libero/libero/bddl_files'),init_states_folder=str(ROOT/'libero/libero/init_files'),
        tsr_root=str(run),tsr_report=str(OUT/('seed%d'%seed)),tsr_recovery=str(run/'recovery/latest.pt'),
        tsr_protocol=str(OUT/'PROTOCOL.json'),tsr_epsilon=.5,p1_enabled=False,e1_enabled=False,g1_enabled=False,
        ifa_interaction_logging=True,ifa_interaction_log_dir=str(run/'ifa_interaction'),
        use_wandb=False,repro_arm='V1_OBJECT_TSR',training=True)
    return cfg


def code_hashes():
    files=list((ROOT/'v1_object_tsr').glob('*.py'))+list((ROOT/'v1_object_tsr').glob('*.sh'))
    files += [ROOT/p for p in json.loads((ROOT/'reports/e1_object_v1/EXPERIMENT_CONTEXT.json').read_text())['files']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PROTOCOL.json').exists(),'Do not overwrite frozen protocol'
    checks=json.loads((OUT/'ENGINEERING_CHECK.json').read_text());assert checks['status']=='PASS'
    assert checks['details']['source_files']==code_hashes(),'CODE_CHANGED_AFTER_SMOKE'
    old=json.loads((ROOT/'reports/e1_object_v1/PROTOCOL.json').read_text())
    fm=ROOT/'reports/e1_object_v1/FEATURE_MANIFEST.json'
    assert sha(fm)==old['feature_manifest_sha256'] and sha(old['checkpoint'])==old['checkpoint_sha256']
    manifest=json.loads(fm.read_text())['42']
    assert all(Path(r['path']).is_file() and Path(r['path']).stat().st_size==r['size'] for rows in manifest.values() for r in rows)
    cfg=config(42)
    checkpoints=[ROOT/('runs/final_paper_text_faithful/ifa/seed42/run_001/mlrifa_%d_model_ep99.pth'%t) for t in range(6,10)]
    assert all(p.is_file() for p in checkpoints)
    write(OUT/'INPUT_MANIFEST.json',dict(feature_manifest=str(fm),sha256=sha(fm),seed=42,
        validation='previous content-hash evidence reused; current existence and sizes verified',
        pretrain=old['checkpoint'],pretrain_sha256=old['checkpoint_sha256'],
        baseline_checkpoints={str(p):sha(p) for p in checkpoints},
        original_window_trajectory_ids='not stored in .pt; no fabricated demonstration IDs'))
    write(OUT/'PROTOCOL.json',dict(name='V1-OBJECT-TaskSelectiveReplay',seed=42,tasks=[6,7,8,9],epochs=100,
        user_protocol_sha256=sha(ROOT/'v1_object_tsr/USER_PROTOCOL.md'),
        approved_adaptation='actual baseline: empty initial replay; 0/1/2/3 old tasks; original 50% window cache; 100 signature windows per planned cache, no base-task replay',
        tau=.1,epsilon=.5,numeric_epsilon=1e-12,signature_batches=10,signature_batch_size=10,
        signature_reference=old['checkpoint'],reference_sha256=old['checkpoint_sha256'],signature_precision='FP32 accumulation then global normalization',
        signature_parameter_space='last temporal decoder block plus complete policy head',signature_mode='eval, dropout disabled, original GMM NLL',
        replay='native DistributedSampler slots and current IDs unchanged; each old slot routed; within each task uniform shuffled cycles without replacement; no importance weights',
        rng='separate local signature/source/within-task streams; cache uses original Random(seed); global training RNG restored after probes',
        updates_per_epoch={'6':157,'7':227,'8':309,'9':362},total_updates=105500,
        baseline_results={'42':old['baseline_results']['42']},baseline_protocol=old['baseline_protocol'],
        baseline_config_sha256=sha(ROOT/'runs/final_paper_text_faithful/ifa/seed42/run_001/config.json'),
        parameter_manifest_sha256=sha(OUT/'PARAMETER_MANIFEST.json'),implementation_files=code_hashes(),
        evaluation=dict(primary_trials_per_arm=200,final_base_task_trials_per_arm=120,total_trials=640,
            arms=['v1','baseline_saved_checkpoints'],n_eval=20,initial_state_ids=list(range(20)),epoch=99,
            action='native release GMM sampling',main_metrics='original four incremental task AUC/FWT/NBT',
            final_sr='all ten tasks, final task9 ep99',baseline_supplement_reason='historical per-trial results unavailable',
            schedule='inherit eval_in_train=false: evaluate saved ep99 endpoints after training'),
        screening=dict(auc_gain_pp=2,final_all10_delta_min=0,fwt_delta_min=0,nbt_delta_max=1,
            individual_old_task_drop_review_pp=10,comparator='historical unrounded main matrix; paired supplement separately reported',
            automatic_additional_seeds=False,not_a_significance_test=True),
        e1_enabled=False,p1_enabled=False,g1_enabled=False,baseline_training_rerun=False,pretraining_rerun=False,
        feature_regeneration=False,formal_old_experiment_queues_resumed=False))
    write(OUT/'CONFIG_seed42.json',cfg)
    write(OUT/'EXPERIMENT_CONTEXT.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        server_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
        python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        frozen_source_files=code_hashes(),protocol_sha256=sha(OUT/'PROTOCOL.json'),
        authorization='one seed42 V1 four-stage training plus paired original-protocol endpoint evaluation',
        baseline_training_rerun=False,old_experiment_queues_resumed=False))


if __name__=='__main__':main()
