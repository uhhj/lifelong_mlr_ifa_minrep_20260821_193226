import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import asdict
from .bindings import bind
bind()
from .io import write,sha
from libero.lifelong.replay_selection import SelectionConfig

S=Path(os.environ['E1_SOURCE']);O=Path(os.environ['E1_OUT'])

def config(out=None):
    root=Path(out) if out else O
    cfg=json.loads((S/'protective_e1/input_config.json').read_text())
    for key in list(cfg):
        if key.startswith('e1_'):del cfg[key]
    cfg.update(seed=42,pretrain='/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth',
       feature_dir=str(O/'features'),buffer_dir=str(root/'buffer'),experiment_dir=str(root/'run_001'),
       experiment_root=str(root),folder='/data/mlr_ifa/datasets',bddl_folder=str(S/'libero/libero/bddl_files'),
       init_states_folder=str(S/'libero/libero/init_files'),e1_root=str(root),e1_contract=str(O/'CONTRACT.json'),
       e1_selection=asdict(SelectionConfig(pool_weighting='uniform')),repro_arm='E1U_uniform_pool',use_wandb=False)
    return cfg

def main():
    assert (O/'features').resolve() == Path('/data/mlr_ifa/experiments/E1_protective_object_seed42_20261008/features').resolve()
    assert O.name == 'E1U_uniform_pool_seed42_20261009' and S.parent.name == 'E1U_uniform_pool_20261009'
    import torch
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    historical=json.loads((S/'protective_e1/historical_feature_manifest.json').read_text())['42']
    actual={};count=0
    for task in (6,7,8,9):
        expected={Path(r['path']).name:r['sha256'] for r in historical[str(task)]}
        files={p.name:p for p in (O/'features'/str(task)).glob('*.pt')}
        assert set(files)==set(expected),(task,len(files),len(expected))
        actual[str(task)]={n:sha(p) for n,p in files.items()}
        assert actual[str(task)]==expected,'CACHE_BYTES_DIFFER_FROM_E0'
        count+=len(files)
    assert count==5921
    write(O/'FEATURE_PARITY.json',dict(status='PASS',all_bytes_match_historical_E0=True,files=count,
       counts={k:len(v) for k,v in actual.items()},historical_manifest_sha256=sha(S/'protective_e1/historical_feature_manifest.json'),features=actual))
    checkpoint='/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth'
    assert sha(checkpoint)=='159351b1df166dd5391c6b1e1317c0ab299fc8b1447fe466acef14ed7896bb7b'
    contract=dict(name='E1U_uniform_pool',source_commit='e963d8d0f3a4626365d33bdb5c83a60821fff115',
      diagnostic_commit='d9a355b6fb75d22ece19cb9c324201e2dae55540',
      sole_algorithm_change='softmax pool weights replaced by exact uniform weights; scoring and random.choices retained',
      device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
      e1_readonly_root='/data/mlr_ifa/experiments/E1_protective_object_seed42_20261008',
      readonly_e1_manifest_sha256=sha(O/'E1_READONLY_BEFORE.json'),baseline='final_paper_text_faithful/ifa/seed42',
      seed=42,feature_parity=sha(O/'FEATURE_PARITY.json'),pretrain_sha256=sha(checkpoint),
      source_sha256={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.yaml','.json') and '.hydra' not in p.parts},
      settings=asdict(SelectionConfig(pool_weighting='uniform')),tasks=[6,7,8,9],n_epochs=100,train_batch_size=10,
      primary=dict(checkpoint_epoch=99,cells=10,trials_per_cell=20),
      secondary=dict(candidate_epochs=[70,75,80,85,90,95,99],cells=34,trials_per_cell=20),
      primary_endpoint='task-balanced overall AUC',extension_rule='strictly positive unrounded overall AUC difference, then fixed seeds43/44; no baseline retraining',
      disclosed_gaps=['Historical baseline ran each seed on one RTX4090; current host has one RTX3090. Precision/library version and GPU count preserved; architecture differs.',
      'Original stage-buffer snapshots unavailable. Same byte-identical feature files, same deterministic E0 buffer implementation and seed reconstruct membership; actual per-stage hashes will be recorded.',
      'E1U reuses the already verified E1 features read-only, with every SHA256 checked against historical E0. No E1U feature export. E1 stage buffer snapshots are directly available; E0 missing historical snapshots limitation remains.'],
      created_unix=time.time(),baseline_rerun=False)
    target=O/'CONTRACT.json'
    if target.exists():
        old=json.loads(target.read_text());contract['created_unix']=old['created_unix'];assert old==contract
    else:write(target,contract)
    write(O/'resolved_training.json',config())
    print(json.dumps(dict(status='PREPARED',feature_files=count,byte_parity=True,contract_sha256=sha(target))),flush=True)

if __name__=='__main__':main()
