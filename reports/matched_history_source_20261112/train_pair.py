"""Task9-only training with the original DEBR gate and one shared probe bank."""
import argparse

from runtime import *
from fork_tools import choose, directional_diagnostic, probe as original_probe, setup_probes
import gate_admission
from updates import update
import torch.distributed as dist


def common_probe(agent, probes, _training_memory):
    return original_probe(agent, probes, probes['shared_memory'])


def initialize(arm, ds):
    agent = make(arm)
    loader, sampler = setup(agent, ds)
    origin = CASE_SPEC['common_initial']
    assert sha(origin['path']) == origin['sha256']
    cp = torch.load(origin['path'], map_location='cpu')
    agent.policy.load_state_dict(cp['state_dict'], strict=True)
    agent.step = origin['global_step']
    agent.optimizer.zero_grad(set_to_none=True)
    agent.policy.train()
    control_seed(SEED)
    return agent, loader, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=list(ARMS))
    parser.add_argument('phase', choices=['initial', 'train'])
    args = parser.parse_args()
    arm = args.arm
    contract = verify()
    capacity()
    dest = OUTPUT[arm]
    if args.phase == 'train':
        assert not (dest / 'recovery.pth').exists() and not (R / ('GATES_%s.jsonl' % arm)).exists()
        with (R / ('TRAIN_STARTED_%s.json' % arm)).open('x') as stream:
            json.dump(dict(pid=os.getpid(), stage=STAGE, observed=time.time(), contract=contract), stream)
    dist.init_process_group('nccl')
    torch.cuda.set_device(0)
    try:
        ds = Dataset(STAGE)
        agent, loader, sampler = initialize(arm, ds)
        initial = capture(agent)
        if COMMON_INITIAL.exists():
            assert equal(initial, torch.load(COMMON_INITIAL, map_location='cpu'))
        else:
            assert args.phase == 'initial' and arm == 'E'
            save(COMMON_INITIAL, initial)
        binding = dict(common_initial_sha256=sha(COMMON_INITIAL),
                       teacher_checkpoint_sha256=CASE_SPEC['common_initial']['sha256'],
                       dataset_manifest_sha256=sha(O / 'ra_coreset/dataset_manifests/task9.json'),
                       global_step=agent.step, optimizer_calls=0, environments=0)
        if args.phase == 'initial':
            write(R / ('INITIAL_%s.json' % arm), dict(status='PASS_COMMON_ALL_STATE', **binding))
            return
        assert all(js(R / ('INITIAL_%s.json' % a)) == dict(status='PASS_COMMON_ALL_STATE', **binding) for a in ARMS)
        del initial
        matched = js(R / 'MATCHED_MEMORY.json')
        assert matched['status'] == 'PASS_MATCHED_HISTORY_CACHE' and matched['sha256'] == sha(MATCHED_MEMORY)
        memories = torch.load(MATCHED_MEMORY, map_location='cpu')
        memory, shared = memories[arm], memories['probe']
        probes = setup_probes(ds, shared)
        probes['shared_memory'] = shared
        probes['metadata']['shared_cache_sha256'] = matched['sha256']
        p = R / 'PROBES.json'
        if p.exists():
            assert js(p) == json.loads(json.dumps(probes['metadata']))
        else:
            write(p, probes['metadata'])
        scale = gate_admission.scale_for(ds)
        info = js(O / 'ra_coreset/ifa_task9.json')['info']
        gate_admission.STAGE = STAGE
        gate_admission.probe = common_probe
        admission = gate_admission.admission(agent, ds, memory, scale, info, probes, 'G')
        assert admission['optimizer_calls'] == 29
        write(R / ('ADMISSION_%s.json' % arm), admission)
        expected = CASE_SPEC['accepted_per_arm']
        assert 100 * len(loader) == expected
        history, gradients = [], []
        blocks = accepted = physical = 0
        for epoch in range(100):
            capacity()
            sampler.set_epoch(epoch)
            order = list(sampler)
            agent.policy.train()
            tic = time.monotonic()
            iterator = iter(loader)
            batch_index = 0
            epoch_rows = []
            measured = False
            while True:
                block = []
                for _ in range(10):
                    item = next(iterator, None)
                    if item is None:
                        break
                    block.append(item)
                if not block:
                    break
                if epoch in (0, 9, 49, 99) and not measured:
                    diagnostic = directional_diagnostic(agent, block[0], memory, probes, 'Q')
                    if diagnostic is not None:
                        measured = True
                        gradients.append(dict(epoch=epoch, batch=batch_index, step=agent.step, **diagnostic))
                        write(R / ('GRADIENTS_%s.json' % arm), gradients)
                base = capture(agent)
                crows = [update(agent, batch, 'C', memory, scale, info, STAGE, retain_weight=0.) for batch in block]
                cs = capture(agent)
                cv = common_probe(agent, probes, memory)
                restore(agent, base)
                qrows = [update(agent, batch, 'Q', memory, scale, info, STAGE) for batch in block]
                qs = capture(agent)
                qv = common_probe(agent, probes, memory)
                decision, checks = choose(cv, qv)
                chosen = qs if decision == 'Q' else cs
                restore(agent, chosen)
                assert equal(chosen, capture(agent))
                epoch_rows.extend(qrows if decision == 'Q' else crows)
                with (R / ('GATES_%s.jsonl' % arm)).open('a') as stream:
                    stream.write(json.dumps(dict(epoch=epoch, batch=batch_index, steps=len(block),
                        start_step=base['step'], end_step=agent.step, C=cv, Q=qv, accepted=decision, checks=checks)) + '\n')
                batch_index += len(block)
                physical += 2 * len(block)
                blocks += 1
                accepted += int(decision == 'Q')
                del base, cs, qs, chosen
            assert batch_index == len(loader)
            totals = {key: sum(x[key] for x in epoch_rows) for key in
                      ('valid', 'invalid', 'old_eligible', 'anchor_slots', 'anchor_actual', 'anchor_empty',
                       'loss', 'bc', 'ifa', 'expert_loss', 'expert_route', 'retain_loss', 'retain_route')}
            assert totals['valid'] == VALID[STAGE] and totals['invalid'] == POP[STAGE] - VALID[STAGE]
            assert totals['old_eligible'] == OLD[STAGE]
            row = dict(epoch=epoch, stage=STAGE, batches=batch_index, global_step=agent.step,
                       physical_optimizer_calls=physical, gate_blocks=blocks, gate_Q_accepts=accepted,
                       seconds=time.monotonic()-tic, totals=totals,
                       sampler_sha256=hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest(),
                       training_probe=common_probe(agent, probes, memory))
            history.append(row)
            state = capture(agent)
            save(dest / 'recovery.pth', dict(state=state, state_dict=state['model'], stage=STAGE,
                next_epoch=epoch+1, history=history, contract=contract, arm=arm,
                physical_optimizer_calls=physical, matched_cache_sha256=matched['sha256'], **binding))
            del state
            write(R / ('EPOCHS_%s.json' % arm), history)
            print(json.dumps(dict(arm=arm, **row)), flush=True)
        assert agent.step - binding['global_step'] == expected and physical == CASE_SPEC['physical_optimizer_per_arm']
        verify()
        checkpoint = model_path(arm, STAGE)
        os.link(dest / 'recovery.pth', checkpoint)
        write(R / ('TRAIN_%s.json' % arm), dict(status='PASS_EXECUTION_PENDING_FINAL', epochs=len(history),
              accepted_updates=expected, physical_optimizer_calls=physical, gate_blocks=blocks,
              gate_Q_accepts=accepted, sha256=sha(checkpoint), checkpoint=str(checkpoint)))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
