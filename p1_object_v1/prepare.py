"""Reuse the already checked OBJECT baseline and feature manifest."""
import json
import platform
import subprocess
from pathlib import Path
from .io import sha,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/p1_object_v1'


def config(seed,run_root=None):
    cfg=json.loads((ROOT/('runs/final_paper_text_faithful/ifa/seed%d/run_001/config.json'%seed)).read_text())
    run=Path(run_root) if run_root else ROOT/('runs/p1_object_v1/seed%d'%seed)
    cfg.update(seed=seed,pretrain=str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        feature_dir=str(ROOT/('features/object_seed%d'%seed)),buffer_dir=str(run/'buffer'),
        experiment_dir=str(run/'checkpoints'),experiment_root=str(run),folder=str(ROOT/'data'),
        bddl_folder=str(ROOT/'libero/libero/bddl_files'),init_states_folder=str(ROOT/'libero/libero/init_files'),
        p1_root=str(run),p1_report=str(OUT/('seed%d'%seed)),p1_recovery=str(run/'recovery/latest.pt'),
        p1_protocol=str(OUT/'PROTOCOL.json'),p1_enabled=True,p1_beta=.05,e1_enabled=False,
        use_wandb=False,repro_arm='P1_OBJECT_v1',training=True)
    return cfg


def code_hashes():
    files=list((ROOT/'p1_object_v1').glob('*.py'))+list((ROOT/'p1_object_v1').glob('*.sh'))
    files += [ROOT/'e1_object_v1'/n for n in ['control.py','runtime.py','io.py','bindings.py','baseline_loop.py']]
    files += [ROOT/p for p in json.loads((ROOT/'reports/e1_object_v1/EXPERIMENT_CONTEXT.json').read_text())['files']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PROTOCOL.json').exists(),'Existing frozen protocol: do not overwrite'
    source=ROOT/'reports/e1_object_v1/PROTOCOL.json';old=json.loads(source.read_text())
    fm=ROOT/'reports/e1_object_v1/FEATURE_MANIFEST.json'
    assert sha(fm)==old['feature_manifest_sha256']
    assert sha(old['checkpoint'])==old['checkpoint_sha256']
    # Reuse verified feature contents; validate expected input sizes/paths now.
    manifest=json.loads(fm.read_text())
    assert all(Path(r['path']).is_file() and Path(r['path']).stat().st_size==r['size'] for tasks in manifest.values() for rows in tasks.values() for r in rows)
    write(OUT/'INPUT_MANIFEST.json',dict(source=str(fm),sha256=sha(fm),validation='previously content-hashed; existence and size rechecked',
        checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256']))
    write(OUT/'PROTOCOL.json',dict(name='P1-OBJECT-v1',user_protocol_sha256=sha(ROOT/'p1_object_v1/USER_PROTOCOL.md'),
        tasks=[6,7,8,9],epochs=100,seed_order=[42,43,44],baseline_results=old['baseline_results'],
        baseline_protocol=old['baseline_protocol'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        e1_enabled=False,beta=.05,mc_samples=16,teacher_batch_size=64,teacher_source='own_task_ep99',teacher_refresh=False,
        teacher_dtype='float32',normalization='mean 16 MC then mean old windows; joint 7D log_prob',
        action_clipping=False,kl_clamping=False,distill_rng='CPU Generator(2000003+seed), continuous across four stages',
        native_forward='unchanged',student_forward='deterministic, gradients enabled',initial_buffer='empty',
        primary_trials=200,primary_checkpoint=99,secondary_evaluation=False,reference_auc=old['reference_auc'],
        append_seeds_if='unrounded seed42 primary AUC > same-seed historical IFA AUC',
        baseline_rerun=False,pretraining_rerun=False,feature_regeneration=False,
        implementation_files=code_hashes()))
    for seed in [42,43,44]:write(OUT/('CONFIG_seed%d.json'%seed),config(seed))
    write(OUT/'EXPERIMENT_CONTEXT.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        server_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
        python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        frozen_source_files=code_hashes(),protocol_sha256=sha(OUT/'PROTOCOL.json'),
        authorization='P1 seed42 four stages and primary evaluation; conditional seeds43/44; no old experiment queues'))


if __name__=='__main__':main()
