"""Migrate a verified formal epoch boundary; never a diagnostic snapshot."""
import json, shutil, subprocess, time
from pathlib import Path
from .common import *

def main():
    import torch
    from cb_ifa_experiment.gpu_runtime import atomic_save,digest
    parent_run=ROOT/'runs/sample_qp_lifelong_v1_continuation_v3/seed46/sample_qp'
    index=read(parent_run/'RECOVERY_INDEX.json')
    assert index['task']==8 and index['epoch']==10 and index['global_step']==41799
    parent=Path(index['path'])
    assert parent==Path('/root/sample_qp_lifelong_v1_continuation_v3/recovery/slot0.pt')
    assert sha(parent)==index['sha256']
    saved=torch.load(parent,map_location='cpu')
    assert (saved['task'],saved['next_epoch'],saved['global_step'])==(8,11,41799)
    assert saved['state']['optimizer_step']==saved['state']['scheduler_step']==3399
    assert len(saved['accepted_epochs'])==211
    assert saved['accepted_epochs'][-1]['epoch']==10
    parent_protocol=ROOT/'reports/sample_qp_lifelong_v1_continuation_v3/PROTOCOL_FROZEN.json'
    assert saved['protocol_sha256']==sha(parent_protocol)
    for name,key in [('task8.json','plan_sha256'),('references_task8.json','reference_sha256')]:
        assert sha(OUT/'plans'/name)==saved[key]
    assert not RUN.exists() and not (STORAGE/'recovery').exists()
    for row in saved['accepted_epochs']:
        assert sha(row['event_path'])==row['event_sha256']
    source_digest=digest(saved['state'])
    initial_protocol_sha=saved['protocol_sha256']
    preflight=read(OUT/'PREFLIGHT_COMPLETE.json')
    assert preflight['V4_PREFLIGHT_PASS'] and all(preflight['gates'].values())
    protocol=read(OUT/'PROTOCOL_FROZEN.json')
    assert sha(OUT/'PROTOCOL_FROZEN.json')==preflight['frozen_protocol_sha256']
    # Actual runtime families, frozen config and evaluation assets only.
    exact=['cb_ifa_experiment/__init__.py','cb_ifa_experiment/gpu_runtime.py',
      'cb_ifa_experiment/common.py','cb_ifa_experiment/real_policy_cpu.py',
      'functional_geometry_diagnostic_v1/__init__.py',
      'functional_geometry_diagnostic_v1/geometry.py',
      'functional_geometry_diagnostic_v1/runtime_guard.py',
      'gradient_relation_experiment/kernel.py','optimizer_signal_diagnostic/core.py',
      'fixed_kappa_experiment/metrics.py']
    names=set(exact)
    for name in protocol['files']:
        if name.startswith('libero/') and not name.endswith(('.hdf5','.pt')):names.add(name)
        if name.startswith('reports/sample_qp_lifelong_v1_continuation_v4/plans/'):names.add(name)
        if name.startswith('checkpoints/') and name.endswith('.pth'):names.add(name)
    for name in names:
        assert name in protocol['files'],('Unfrozen dependency',name)
        assert sha(ROOT/name)==protocol['files'][name],('Parent runtime changed',name)
    numeric=[*Path('sample_qp_optimized_v1_continuation_v4').glob('*.py'),
             Path('sample_qp_lifelong_v1_continuation_v4/step.py')]
    for f in numeric:assert sha(ROOT/f)==protocol['files'][str(f)],str(f)
    files={name:protocol['files'][name] for name in sorted(names)}
    for folder in ('sample_qp_optimized_v1_continuation_v4','sample_qp_lifelong_v1_continuation_v4'):
        for f in sorted((ROOT/folder).glob('*.py')):files[str(f.relative_to(ROOT))]=sha(f)
    files['.migration/activate.sh']=sha(ROOT/'.migration/activate.sh')
    execution=dict(version='v4_epoch_continuation_execution_v1',
        scientific_protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),
        preflight_sha256=sha(OUT/'PREFLIGHT_COMPLETE.json'),files=files,
        numerical_sources_unchanged=True,
        cache_integrity='Original CachedWindows/reference SHA256 on consumption; no whole-repository hash',
        epoch11_observation={'185':'LBFGSB PASS, trust not called','258':'trust PASS',
          'commit':'reload exact canonical state at epoch11 boundary, then continue'},
        stop_on_any_training_failure=True,automatic_restart=False,
        seed=46,tasks=[8,9],epochs_per_task=100,eval=protocol['eval'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    assert not (OUT/'EXECUTION_FROZEN.json').exists()
    write(OUT/'EXECUTION_FROZEN.json',execution)
    # Four recovery-sized allocations cover parent copy, two rotating slots, and atomic temp.
    checkpoint_bytes=Path('/root/sample_qp_lifelong_v1/checkpoints/mlrifa_6_model_ep99.pth').stat().st_size
    root_need=4*parent.stat().st_size+14*checkpoint_bytes+4*2**30
    data_need=256*2**20+1*2**30
    disk=dict(root_free=shutil.disk_usage('/root').free,data_free=shutil.disk_usage(ROOT).free,
              root_required_with_reserve=root_need,data_required_with_reserve=data_need)
    assert disk['root_free']>root_need and disk['data_free']>data_need,disk
    RUN.mkdir(parents=True);(RUN/'events').mkdir()
    (STORAGE/'parents').mkdir(parents=True);(STORAGE/'recovery').mkdir();(STORAGE/'checkpoints').mkdir()
    copied=STORAGE/'parents/v3_epoch010_slot0.pt'
    shutil.copyfile(parent,copied);assert sha(copied)==index['sha256'];copied.chmod(0o444)
    saved['parent_recovery']=dict(path=str(parent),sha256=index['sha256'],protocol_sha256=initial_protocol_sha)
    saved['protocol_sha256']=sha(OUT/'PROTOCOL_FROZEN.json')
    saved['execution_protocol_sha256']=sha(OUT/'EXECUTION_FROZEN.json')
    child=STORAGE/'recovery/slot0.pt';atomic_save(child,saved)
    loaded=torch.load(child,map_location='cpu');assert digest(loaded['state'])==source_digest
    assert loaded['accepted_epochs']==saved['accepted_epochs']
    write(RUN/'RECOVERY_INDEX.json',dict(path=str(child),sha256=sha(child),task=8,epoch=10,global_step=41799))
    write(RUN/'ACCEPTED_EPOCHS.json',saved['accepted_epochs'])
    for task in (6,7):
        name='TASK%d_COMPLETE.json'%task;stage=read(parent_run/name)
        assert sha(stage['checkpoint'])==stage['sha256']
        shutil.copyfile(parent_run/name,RUN/name)
        (STORAGE/'checkpoints'/('mlrifa_%d_model_ep99.pth'%task)).symlink_to(stage['checkpoint'])
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader']).decode().strip()
    write(RUN/'ATTEMPT_000.json',dict(kind='PARENT_CHILD_MIGRATION_NOT_TRAINING',
        gpu_identity=identity,time=time.time(),parent_index=index,
        v4_gpu_parity_evidence=str(OUT/'PREFLIGHT_COMPLETE.json')))
    report=dict(status='PASS',parent=index,parent_copy=str(copied),child=str(child),child_sha256=sha(child),
        task=8,next_epoch=11,global_step=41799,optimizer_step=3399,scheduler_step=3399,
        state_digest=source_digest,complete_state_unchanged=True,accepted_epochs=211,
        diagnostic_snapshots_used=False,source_unchanged=sha(parent)==index['sha256'],disk=disk,
        protocol_sha256=saved['protocol_sha256'],execution_sha256=saved['execution_protocol_sha256'],
        formal_training_resumed=False)
    write(OUT/'EPOCH_RECOVERY_MIGRATION.json',report)
    shutil.copyfile(OUT/'FINAL_REPORT.md',OUT/'PREFLIGHT_REPORT.md')
    write(OUT/'STATUS.json',dict(status='EPOCH_MIGRATION_READY',formal_training_resumed=False))
    print(json.dumps(report,allow_nan=False),flush=True)

if __name__=='__main__':main()
