"""Post-completion CPU reconstruction; zero optimizer updates and environments.

Run only as: python -B Q/final_case_cpu.py OBJECT_ALIGN_41
The wrapper must first prove the new controller's natural successful exit and
absence of own scientific processes. This file is prepared, not executed.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys, json, hashlib, random, io, contextlib, math, copy, time
from pathlib import Path
Q = Path(__file__).resolve().parent
assert len(sys.argv) == 2
CASE = sys.argv[1]
assert CASE in ['%s_%s_%d' % (s, v, seed) for s in ('OBJECT', 'GOAL') for v in ('ALIGN', 'MLR') for seed in (41, 42, 43)]
sys.path.insert(0, str(Q / 'cases' / CASE))
from case_runtime import *
from libero.lifelong.models.brief import BriefPolicy


def finite(x):
    if torch.is_tensor(x):
        assert torch.isfinite(x).all()
    elif isinstance(x, dict):
        for v in x.values():
            finite(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            finite(v)


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def linear(step, total):
    if step < 500:
        return float(step) / 500
    return max(0., float(total - step) / max(1, total - 500))


class Wrapped(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.module = BriefPolicy(copy.deepcopy(cfg), None)


def construct(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return Wrapped(cfg)


def infer(model, feature):
    if feature.ndim == 5:
        feature = feature.squeeze(1)
    assert feature.shape[1:] == (8, 4, 512)
    with torch.no_grad():
        h = model.module.temporal_encode(feature)[:, -1]
        result = model.module.policy_head.forward_fn(h)
    finite(result)
    return tuple(x.detach().cpu().clone() for x in result)


def main():
    tic = time.time()
    assert INPUT['case'] == CASE and INPUT['variant'] in ('ALIGN', 'MLR')
    assert (Q / 'controller.exit').read_text().strip() == '0', 'No final CPU before natural controller success'
    assert (R / 'train_A.exit').read_text().strip() == '0'
    assert (R / 'evaluate.exit').read_text().strip() == '0'
    contract = verify()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg = EasyDict(js(R / 'resolved_random.json'))
    cfg.device = 'cpu'
    variant = INPUT['variant']
    budget = 105500 if SUITE == 'OBJECT' else 76100
    assert INPUT['production_budget'] == budget and INPUT['admission_optimizer_calls'] == 20
    expected_objective = ('ALIGN_BC' if variant == 'ALIGN' else 'ORIGINAL_BC') + ' + 0.1*IFA'
    hist = js(R / 'EPOCHS_A.json')
    assert len(hist) == 400
    train = js(R / 'TRAIN_A.json')
    assert train['status'] == 'PASS_EXECUTION_PENDING_FINAL'
    assert train['variant'] == variant and train['objective'] == expected_objective
    assert train['epochs'] == 400 and train['steps'] == train['physical_optimizer_calls'] == budget
    assert train['physical_calls_observed'] == dict(production=budget, admission=20)
    assert train['training_collection_episodes'] == 0
    initial = js(R / 'INITIAL_A.json')
    assert initial['status'] == 'PASS_OFFICIAL_INITIALIZATION'
    assert initial['tensors'] == INPUT['initial_state']['tensors']
    assert all(initial['rng'][k] == INPUT['initial_state']['rng'][k] for k in initial['rng'] if k != 'cuda')
    assert initial['official_cp_sha256'] == sha(cfg.pretrain)
    assert initial['historical_cuda_rng_equal'] == (initial['rng']['cuda'] == INPUT['initial_state']['rng']['cuda'])
    checkpoints, admissions, stage_checks = {}, {}, {}
    cumulative = 0
    model_forwards = 0
    model_windows = 0
    final_model = None
    for stage in range(6, 10):
        ds = Dataset(stage)
        batches = (POP[stage] + 9) // 10
        total = batches * 100
        target = model_path('A', stage)
        raw = torch.load(target, map_location='cpu')
        sd = raw['state_dict']
        finite(sd)
        assert raw['stage'] == stage and raw['contract'] == contract
        assert raw['steps'] == cumulative + total
        assert raw['manifest_sha256'] == sha(R / 'manifests' / ('task%d.json' % stage))
        assert raw['ifa_sha256'] == sha(R / ('ifa_task%d.json' % stage))
        if stage < 9:
            assert raw['epoch'] == 99
        else:
            assert raw['next_epoch'] == 100
        model = construct(cfg)
        assert len(sd) == len(model.state_dict()) == 100
        model.load_state_dict(sd, strict=True)
        assert equal(model.state_dict(), sd)
        model.eval()
        # A CPU serialization/load/replay check, not a historical GPU-output claim.
        stream = io.BytesIO()
        torch.save(sd, stream)
        stream.seek(0)
        reloaded = torch.load(stream, map_location='cpu')
        assert equal(reloaded, sd)
        replay_model = construct(cfg)
        replay_model.load_state_dict(reloaded, strict=True)
        replay_model.eval()
        sample = torch.utils.data.default_collate([ds[0], ds[len(ds)-1]])[0]['feature']
        before = cpu(rng())
        values = infer(model, sample)
        repeated = infer(replay_model, sample)
        assert equal(values, repeated) and equal(before, cpu(rng()))
        model_forwards += 2
        model_windows += 4
        cp_sha = sha(target)
        checkpoints[str(stage)] = dict(sha256=cp_sha, tensor_keys=100, strict_load=True,
            finite=True, cpu_serialization_exact=True, cpu_parameter_forward_replay_exact=True,
            replay_windows=2, model_forward_calls=2, historical_GPU_output_equality_claimed=False)
        assert train['checkpoints'][str(stage)] == cp_sha
        stage_record = js(R / ('STAGE_A_%d.json' % stage))
        assert stage_record['sha256'] == cp_sha and stage_record['steps'] == cumulative + total
        binding = js(R / ('TRAIN_INPUT_A_%d.json' % stage))
        expected_valid = VALID[stage] if variant == 'ALIGN' else POP[stage]
        assert binding['variant'] == variant and binding['objective'] == expected_objective
        assert binding['bc_valid'] == expected_valid and binding['bc_invalid'] == POP[stage]-expected_valid
        assert binding['source_alignment_label_valid'] == VALID[stage]
        assert not any(binding[k] for k in ('execution_expert', 'retention', 'support_weights', 'candidate_gating'))
        assert binding['online_training_collection'] == 0
        assert binding['manifest_sha256'] == raw['manifest_sha256'] and binding['ifa_sha256'] == raw['ifa_sha256']
        schedule = js(R / ('SCHEDULE_A_%d.json' % stage))
        assert schedule['status'] == 'PASS' and schedule['total_updates'] == total and schedule['warmup'] == 500
        ad = js(R / ('ADMISSION_A_%d.json' % stage))
        assert ad['status'] == 'PASS_OWN_GPU_STRIPPED_S_ADMISSION' and ad['variant'] == variant
        assert ad['stage'] == stage and ad['arm'] == 'A' and ad['contract'] == contract
        assert ad['optimizer_calls'] == 5 and ad['extra_autograd_calls'] == ad['production_calls_added'] == ad['environments'] == 0
        assert all(ad[k] for k in ('direct_bc_ifa_first_update_full_state_exact',
            'direct_bc_ifa_second_update_full_state_exact', 'roundtrip_exact', 'next_step_exact', 'restore_initial_exact'))
        assert ad['empty_mask_differentiable_zero'] is True if variant == 'ALIGN' else ad['mlr_ignores_alignment_validity'] is True
        admissions[str(stage)] = dict(sha256=sha(R / ('ADMISSION_A_%d.json' % stage)), optimizer_calls=5, actual_stage_pass=True)
        rows = hist[(stage-6)*100:(stage-5)*100]
        for ep, row in enumerate(rows):
            steps = cumulative + (ep+1)*batches
            assert row['stage'] == stage and row['epoch'] == ep and row['batches'] == batches
            assert row['steps'] == row['physical_optimizer_calls'] == steps
            assert row['physical_calls_observed'] == dict(production=steps, admission=(stage-5)*5)
            assert row['totals']['valid'] == expected_valid and row['totals']['invalid'] == POP[stage]-expected_valid
            assert all(math.isfinite(float(v)) for v in row['totals'].values())
            order = torch.randperm(POP[stage], generator=torch.Generator().manual_seed(ep)).numpy().astype(np.int64)
            assert digest_array(order) == row['sampler_sha256']
            base_lr = cfg.train.optimizer.kwargs.lr
            assert math.isclose(row['lr_first'], base_lr*linear(ep*batches,total), rel_tol=1e-12, abs_tol=1e-15)
            assert math.isclose(row['lr_after'], base_lr*linear((ep+1)*batches,total), rel_tol=1e-12, abs_tol=1e-15)
        stage_checks[str(stage)] = dict(epochs=100, batches_per_epoch=batches, population=POP[stage],
            effective_valid=expected_valid, optimizer_updates=total, random_order_and_lr_recomputed=True,
            manifest_sha256=raw['manifest_sha256'], ifa_sha256=raw['ifa_sha256'])
        cumulative += total
        if stage == 9:
            final_model = model
        del ds, replay_model, stream, reloaded, raw, sd, values, repeated, sample
    assert cumulative == budget
    rec_path = OUTPUT['A'] / 'recovery.pth'
    rec = torch.load(rec_path, map_location='cpu')
    state = rec['state']
    assert set(state) == {'model','optimizer','scheduler','grads','modes','requires','rng','step'}
    assert rec['history'] == hist and rec['stage'] == 9 and rec['next_epoch'] == 100
    assert rec['contract'] == contract and rec['steps'] == rec['physical_optimizer_calls'] == state['step'] == budget
    assert equal(rec['state_dict'], state['model']) and equal(state['model'], final_model.state_dict())
    assert rec_path.stat().st_ino == model_path('A', 9).stat().st_ino
    finite(state['model']); finite(state['optimizer']); finite(state['grads'])
    assert state['optimizer']['state'] and state['scheduler']
    for name, parameter in final_model.named_parameters():
        parameter.requires_grad_(state['requires'][name])
        parameter.grad = None if state['grads'][name] is None else state['grads'][name].clone()
    for name, module in final_model.named_modules():
        module.training = state['modes'][name]
    optimizer = torch.optim.AdamW(final_model.parameters(), **dict(cfg.train.optimizer.kwargs))
    optimizer.load_state_dict(state['optimizer'])
    total9 = ((POP[9]+9)//10)*100
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: linear(step,total9))
    scheduler.load_state_dict(state['scheduler'])
    optimizer.load_state_dict(state['optimizer'])
    assert equal(optimizer.state_dict(), state['optimizer']) and equal(scheduler.state_dict(), state['scheduler'])
    assert scheduler.last_epoch == total9
    for saved in state['optimizer']['state'].values():
        assert int(saved['step']) == total9
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy']); torch.set_rng_state(state['rng']['cpu'])
    assert equal(random.getstate(),state['rng']['python']) and equal(np.random.get_state(),state['rng']['numpy'])
    assert torch.equal(torch.get_rng_state(),state['rng']['cpu'])
    assert len(state['rng']['cuda']) == 1 and state['rng']['cuda'][0].dtype == torch.uint8
    assert all(equal(p.grad,state['grads'][n]) and p.requires_grad==state['requires'][n] for n,p in final_model.named_parameters())
    assert {n:m.training for n,m in final_model.named_modules()} == state['modes']
    eval_ad = js(R / 'EVAL_ADMISSION.json')
    assert eval_ad['status'] == 'PASS' and eval_ad['environments'] == eval_ad['updates'] == 0
    assert [x['length'] for x in eval_ad['cache_parity']] == list(range(1,9))
    assert eval_ad['official_cp_sha256'] == initial['official_cp_sha256']
    complete = js(R / 'EVALUATION_COMPLETE.json')
    assert complete['episodes'] == 200 and complete['vectors'] == 20 and complete['cells'] == 10
    records = js(R / 'OUTCOMES.json')
    assert len(records) == 200
    from libero.libero.benchmark import get_benchmark
    with contextlib.redirect_stdout(io.StringIO()):
        benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    initials = {}
    for task in range(6,10):
        task_info = benchmark.get_task(task)
        path = Path(cfg.init_states_folder) / task_info.problem_folder / task_info.init_states_file
        states = torch.load(path, map_location='cpu')
        initials[task] = dict(file_sha256=sha(path), hashes=[digest_array(np.asarray(x)) for x in states[:20]])
    expected_keys = {(stage,task,i) for stage in range(6,10) for task in range(6,stage+1) for i in range(20)}
    seen = set()
    for record in records:
        stage,task,i = record['stage'],record['task'],record['initial_index']
        key = (stage,task,i)
        assert key in expected_keys and key not in seen; seen.add(key)
        assert record['arm'] == 'A' and record['attempt'] == 1 and record['env_seed'] == SEED+i//10
        assert record['cp_sha256'] == checkpoints[str(stage)]['sha256']
        assert record['initial_file_sha256'] == initials[task]['file_sha256'] and record['initial_sha256'] == initials[task]['hashes'][i]
        eps = np.random.RandomState(20261024+SEED*1000+task*100+i).standard_normal((600,7)).astype(np.float32)
        assert record['noise_sha256'] == digest_array(eps)
        assert type(record['success']) is bool and 1 <= record['steps'] <= 600
        assert record['termination'] == ('success' if record['success'] else 'timeout600')
        assert record['success'] or record['steps'] == 600
        assert len(record['action_stream_sha256']) == 64
    assert seen == expected_keys
    for stage in range(6,10):
        for task in range(6,stage+1):
            for group in (0,1):
                part = js(R/'outcomes'/('A_s%d_t%d_g%d.json' % (stage,task,group)))
                assert part == [x for x in records if x['stage']==stage and x['task']==task and x['initial_index']//10==group]
    assert verify() == contract and CALLS == dict(production=0, admission=0) and not torch.cuda.is_initialized()
    result = dict(status='PASS_CPU_CASE_RECONSTRUCTION', case=CASE, suite=SUITE, seed=SEED, variant=variant,
        contract=contract, checkpoints=checkpoints, stages=stage_checks, admissions=admissions,
        recovery=dict(sha256=sha(rec_path),model=True,optimizer=True,scheduler=True,grads=True,modes=True,
            requires_grad=True,cpu_rng_exact=True,saved_cuda_rng_preserved_not_executed=True,steps=budget),
        initialized_tensors_and_nonCUDA_rng_exact=True, historical_cuda_rng_equal=initial['historical_cuda_rng_equal'],
        hardware=initial['hardware'], production_optimizer=budget, admission_optimizer=20, training_collection=0,
        episodes=200, initial_states_and_paired_noise_exact=True, actual_attempts_all_one=True,
        model_forward_calls=model_forwards, model_windows=model_windows, optimizer_calls=0,
        extra_autograd_calls=0, environments=0, seconds=time.time()-tic,
        light_sha256={n:sha(R/n) for n in ('INPUT.json','INITIAL_A.json','EPOCHS_A.json','TRAIN_A.json','OUTCOMES.json','TASK6_S_COMPARISON.json')})
    write(R / 'FINAL_CPU.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
