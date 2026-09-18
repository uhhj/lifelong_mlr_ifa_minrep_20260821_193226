"""Freeze native replay membership, stage references and all implementation inputs."""
import json
import math
import random
import subprocess
import time
from pathlib import Path
from .common import *

def main():
    assert not (OUT/'PROTOCOL_FROZEN.json').exists(), 'No protocol overwrite'
    assert not RUN.exists(), 'Existing formal run is preserved'
    OUT.mkdir(parents=True, exist_ok=True); STORAGE.mkdir(exist_ok=True)
    from cb_ifa_experiment import gpu_runtime as runtime
    from libero.lifelong.algos import mlr
    from libero.libero.benchmark import get_benchmark
    feature_root = ROOT/'features/object_seed46'
    hist = ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed46/mlr'
    buffer = STORAGE/'planning_buffer'
    assert not buffer.exists(), 'Planning buffer must start empty'
    buffer.mkdir()
    files = {}; plans = []; native_buffer_events = []
    for task in TASKS:
        ds = mlr.build_dataset_for_task(str(feature_root), task, str(buffer), SEED, start=6)
        records = []
        for source in ds.file_paths:
            path = Path(source); tid = int(mlr.parse_path(str(path)))
            name = path.name.split('__',1)[1] if path.name.startswith('t') else path.name
            origin = feature_root/str(tid)/name
            h = sha(origin); assert sha(path) == h
            records.append(dict(origin=str(origin.relative_to(ROOT)), task=tid, sha256=h,
                                replay=tid<task))
            files[str(origin.relative_to(ROOT))] = h
        old = read(hist/'plans'/('task%d.json' % task))
        assert [r['origin'] for r in records] == old['immutable_feature_origins']
        assert [r['sha256'] for r in records] == old['record_sha256']
        plan = dict(task=task, records=records, batches_per_epoch=math.ceil(len(records)/10),
            epochs=100, sampler=dict(seed=0, num_replicas=1, rank=0, shuffle=True, drop_last=False),
            batch_size=10, num_workers=0, native_historical_membership_exact=True)
        assert plan['batches_per_epoch'] == old['batches_per_epoch']
        write(OUT/('plans/task%d.json' % task), plan)
        write(OUT/('plans/references_task%d.json' % task), selected_references(plan))
        plans.append(dict(task=task, records=len(records), batches=plan['batches_per_epoch'],
                          updates=plan['batches_per_epoch']*100, references=32*(task-6)))
        result = mlr.update_buffer_equal_quota_10(str(feature_root),task,str(buffer),
            capacity=mlr.BUFFER_CAPACITY,seed=SEED,drop_policy='random',rate=.5)
        native_buffer_events.append(dict(task=task,result=result))
    feature_cfg = ROOT/'runs/fixed_kappa_confirmation_v2/feature_seed46/run_001/config.json'
    front = read(feature_cfg)['policy']
    assert front['use_language'] is True and front['use_state'] is False
    assert sum(p['updates'] for p in plans) == 105500
    # Record only observational setup: CPU planning never evaluates a candidate policy.
    bench = get_benchmark('LIBERO_OBJECT')(0)
    assets = [ROOT/'checkpoints/object_multitask_model_ep99.pth',
              hist/'run_001/config.json', feature_cfg]
    for task in range(10):
        obj = bench.get_task(task)
        assets.extend([ROOT/'data'/bench.get_task_demonstration(task),
            ROOT/'libero/libero/bddl_files'/obj.problem_folder/obj.bddl_file,
            ROOT/'libero/libero/init_files'/obj.problem_folder/obj.init_states_file])
    for folder in ['sample_qp_lifelong_v1','sample_qp_optimized_v1','sample_task_qp_v1',
                   'functional_geometry_diagnostic_v1','gradient_relation_experiment',
                   'cb_ifa_experiment','optimizer_signal_diagnostic','fixed_kappa_experiment','libero']:
        assets.extend((ROOT/folder).rglob('*.py'))
    assets.extend((ROOT/'libero/configs').rglob('*.yaml'))
    assets.extend((OUT/'plans').glob('*.json'))
    for f in sorted(set(assets)):
        assert f.is_file(), str(f)
        files[str(f.relative_to(ROOT))] = sha(f)
    protocol = dict(experiment='sample_qp_lifelong_v1',seed=46,seed_status='development',
        reason_seed46='Same-GPU long exact engineering validation; chosen before this full-run outcome',
        arm='SAMPLE_STEP_QP', tasks=list(TASKS), stage_plans=plans,total_updates=105500,
        algorithm='Hard joint per-window QP on actual native AdamW displacement; retain native moments',
        reference_rule='32 retained windows per old task uniformly without replacement by private hash-derived seed; fixed within each stage; gradients refreshed every step',
        replay_rule='Native data-independent buffer update reproduced in advance; exact historical memberships and order; immutable copies of retained serialized windows',
        optimizer='Unchanged AdamW; original per-task reset via start_task; no resets within task',
        loss='Original mixed GMM BC, original weights, original grad clipping and batch-step schedule',
        frontend=dict(use_language=True,use_state=False), checkpoint_epochs=list(SAVED_EPOCHS),
        recovery='Full state after each completed epoch; two alternating snapshots and immutable stage boundaries; no automatic restart or seed replacement',
        eval=dict(entry='unchanged native test.py with passive evidence observer',epochs=[99],
                  trials_per_cell=20,cells=10,total_trials=200,task_order='stage-major then seen task ascending',
                  GMM='native action sampling',success='native done accumulator',select_best=False),
        comparison='Historical MLR seed46 context only, unpaired across hardware. No superiority conclusion.',
        no_interim_stopping=True,no_hyperparameter_search=True,
        decisions=['SINGLE_SEED_PILOT_COMPLETE','EXECUTION_FAILED_OR_INCOMPLETE'],
        resources=dict(training_hours_ceiling=120,evaluation_hours_ceiling=8,
                       cuda_allocated_GiB=20,cuda_reserved_GiB=22,host_rss_GiB=28),
        preflight='CPU math 0/32/64/96; GPU native task6 and joint32/64/96; real legacy64 regression; state recovery and epoch boundary; logging noninterference',
        source_sha=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        freeze_time=time.time(),files=files)
    write(OUT/'ASSET_AUDIT.json',dict(status='PASS',stage_plans=plans,
          buffer_events=native_buffer_events,frontend=protocol['frontend'],file_count=len(files)))
    write(OUT/'PROTOCOL_FROZEN.json',protocol)
    (OUT/'PROTOCOL.md').write_text('# Full lifelong per-sample QP pilot\n\n'+
        'Seed46 only. This is a new controlled developmental pilot, not an independent confirmation.\n\n'+
        'Tasks6→9, 100 epochs each, 105500 updates. Full native mixed BC, clipping, AdamW and schedule. '
        'AdamW is reset at original task boundaries only. Hard joint actual-displacement QP; no kappa, Fisher or loss reweighting.\n\n'+
        'Per-stage fixed private selection of 32 windows per old task from the actual native buffer membership. '
        'All reference gradients refreshed every update; 0/32/64/96 constraints. No extra gradient is added to BC.\n\n'+
        'Native buffer membership/order reconstructed using the unchanged native functions and verified against historical plans. '
        'Window identities/hashes and private sampling seeds are fixed before training.\n\n'+
        'Fixed ep99, original test.py order and sampled GMM actions, 20 trials per cell/200 total. No best checkpoint. '
        'All cells, failures and per-trial successes retained. Historical MLR is unpaired context only.\n\n'+
        'No outcome-based stop or seed replacement. Numerical/resource failure stops with evidence. '
        'Epoch-boundary recovery retains model, moments, scheduler, RNG, modes, sampler cursor, reference membership and accepted event files.\n')
    print('PREPARATION_PASS',plans,flush=True)

if __name__ == '__main__':main()
