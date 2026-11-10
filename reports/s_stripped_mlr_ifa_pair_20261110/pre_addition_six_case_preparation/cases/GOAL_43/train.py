"""Full-S training lifecycle with only MLR+IFA+ALIGN and frozen Random inputs.

Each accepted batch is executed once. No gate alternatives, support memories,
teacher caches, online collection, auxiliary expert or retention losses exist.
"""
import argparse
from case_runtime import *
from updates import update
from admission import admit
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=list(ARMS))
    arm = parser.parse_args().arm
    verify()
    capacity()
    dest = OUTPUT[arm]
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'recovery.pth').exists(), 'No blind resume or completed-run restart'
    assert not (R / ('TRAIN_' + arm + '.json')).exists()
    dist.init_process_group('nccl')
    torch.cuda.set_device(0)
    history = []
    physical = 0
    try:
        a = make(arm)
        initialization(a, arm)
        assert a.step == 0
        for stage in range(6, 10):
            write(R / 'TRAIN_STATUS.json', dict(arm=arm, stage=stage, phase='prepare', time=time.time(), pid=os.getpid()))
            ds = Dataset(stage)
            loader, sampler = setup(a, ds)
            info_path = R / ('ifa_task%d.json' % stage)
            info = js(info_path)['info']
            manifest = R / 'manifests' / ('task%d.json' % stage)
            assert len(ds) == POP[stage]
            assert sum(bool(r['bc_valid']) for r in ds.rows) == VALID[stage]
            assert len(loader) == (len(ds) + 9) // 10
            a.policy.train()
            input_binding = dict(manifest_sha256=sha(manifest), ifa_sha256=sha(info_path))
            write(R / ('TRAIN_INPUT_%s_%d.json' % (arm, stage)), dict(
                status='PASS_FROZEN_RANDOM_INPUT', stage=stage, population=len(ds), batches=len(loader),
                bc_valid=VALID[stage], bc_invalid=POP[stage]-VALID[stage],
                objective='ALIGN_BC + 0.1*IFA', random_members_and_order=True,
                execution_expert=False, retention=False, support_weights=False,
                candidate_gating=False, online_training_collection=0, **input_binding))
            set_count_mode('admission')
            admission_before = CALLS['admission']
            admit(a, ds, info, stage, arm)
            assert CALLS['admission'] - admission_before == 5
            set_count_mode('production')
            for ep in range(100):
                capacity()
                sampler.set_epoch(ep)
                order = list(iter(sampler))
                a.policy.train()
                tic = time.time()
                lr = a.optimizer.param_groups[0]['lr']
                totals = dict(loss=0., bc=0., ifa=0., valid=0, invalid=0)
                batches = 0
                for batch in loader:
                    stats = update(a, batch, info, stage)
                    for key in totals:
                        totals[key] += stats[key]
                    physical += 1
                    batches += 1
                assert batches == len(loader) and CALLS['production'] == physical == a.step
                assert totals['valid'] == VALID[stage] and totals['invalid'] == POP[stage]-VALID[stage]
                row = dict(stage=stage, epoch=ep, steps=a.step, batches=batches,
                    seconds=time.time()-tic, lr_first=lr, lr_after=a.optimizer.param_groups[0]['lr'],
                    sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),
                    totals=totals, physical_optimizer_calls=physical, physical_calls_observed=dict(CALLS))
                history.append(row)
                state = capture(a)
                save(dest / 'recovery.pth', dict(state=state, state_dict=state['model'], stage=stage,
                    next_epoch=ep+1, steps=a.step, contract=verify(), history=history,
                    physical_optimizer_calls=physical, **input_binding))
                write(R / ('EPOCHS_' + arm + '.json'), history)
                write(R / 'TRAIN_STATUS.json', dict(arm=arm, stage=stage, phase='train',
                    epoch_complete=ep+1, steps=a.step, physical_optimizer_calls=physical,
                    time=time.time(),pid=os.getpid()))
                print(json.dumps(dict(arm=arm, **row)), flush=True)
            target = model_path(arm, stage)
            assert not target.exists()
            if stage == 9:
                os.link(dest / 'recovery.pth', target)
            else:
                save(target, dict(state_dict=cpu(a.policy.state_dict()), stage=stage, epoch=99,
                    steps=a.step, contract=verify(), **input_binding))
            write(R / ('STAGE_%s_%d.json' % (arm, stage)), dict(status='PASS_TRAINING_PENDING_FINAL',
                model=str(target), sha256=sha(target), steps=a.step, epochs=100, stage=stage,
                **input_binding))
            if stage == 6:
                # Read-only diagnosis, never a restart/retrain trigger. Full S
                # can include its current execution term already at Task6.
                spec = js(R / 'INPUT.json')
                source = Path(spec['full_s_task6_path'])
                assert sha(source) == spec['full_s_task6_sha256']
                before_rng = cpu(rng())
                reference = torch.load(source, map_location='cpu')['state_dict']
                actual = cpu(a.policy.state_dict())
                assert actual.keys() == reference.keys()
                differences = {}
                for key, value in actual.items():
                    expected_value = reference[key]
                    assert value.shape == expected_value.shape and value.dtype == expected_value.dtype
                    assert torch.isfinite(value).all() and torch.isfinite(expected_value).all()
                    if not torch.equal(value, expected_value):
                        differences[key] = float((value.double()-expected_value.double()).abs().max())
                assert equal(before_rng, cpu(rng()))
                write(R / 'TASK6_S_COMPARISON.json', dict(
                    status='RECORDED_ACTUAL_TENSORS_NOT_A_HISTORICAL_EQUALITY_GATE',
                    source_path=str(source), source_sha256=spec['full_s_task6_sha256'],
                    current_checkpoint_sha256=sha(target), tensor_count=len(actual),
                    all_tensors_exact=not differences, different_tensors= differences,
                    current_device=torch.cuda.get_device_properties(0).name,
                    initialization_evidence='INITIAL_'+arm+'.json',
                    full_s_current_execution_term_removed=True,
                    no_single_cause_attribution_from_tensor_difference=True,
                    no_hardware_equivalence_inferred=True,
                    optimizer_calls=0, policy_forward_calls=0, environments=0))
                del actual, reference, before_rng
        expected = sum(((POP[t]+9)//10)*100 for t in range(6,10))
        assert expected == (105500 if SUITE == 'OBJECT' else 76100)
        assert CALLS == dict(production=expected, admission=20)
        assert a.step == expected and len(history) == 400 and physical == expected
        assert all(sum(r['stage']==t for r in history)==100 for t in range(6,10))
        write(R / ('TRAIN_' + arm + '.json'), dict(status='PASS_EXECUTION_PENDING_FINAL', epochs=400,
            steps=a.step, physical_optimizer_calls=physical, physical_calls_observed=dict(CALLS),
            summed_epoch_seconds=sum(x['seconds'] for x in history), training_collection_episodes=0,
            objective='ALIGN_BC + 0.1*IFA', sampling='frozen Random members and order',
            checkpoints={str(t):sha(model_path(arm,t)) for t in range(6,10)}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
