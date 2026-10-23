"""Read-only local diagnostic recomputation; no model loading or experiment changes."""
from pathlib import Path
import hashlib
import json
import statistics as st

R = Path(__file__).resolve().parent
M = R / 'snapshot_20261023_141307/metadata'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    events = [json.loads(x) for x in (M / 'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    epochs = [x for x in events if x['event'] == 'probe_epoch']
    costs = json.loads((M / 'COST_AUDIT.json').read_text(encoding='utf8'))
    curves = []
    for model in ('P0', 'P1', 'P2', 'P3', 'P4'):
        for task in (6, 7):
            selected = [x for x in costs['costs'] if x['key'].startswith(f'native_{model}_task{task}_')]
            fits = [[e for e in epochs if e['key'] == x['key']] for x in selected]
            assert len(selected) == 5 and all([e['epoch'] for e in f] == list(range(50)) for f in fits)
            curves.append(dict(model=model, task=task,
                selected_epochs=[x['selected_epoch'] for x in selected],
                updates_per_fold=[x['updates'] for x in selected],
                trainable_parameters=selected[0]['trainable_parameters'],
                training_loss_epoch0_mean=st.mean(f[0]['training_loss'] for f in fits),
                training_loss_epoch49_mean=st.mean(f[-1]['training_loss'] for f in fits),
                validation_epoch49_mean_mm=1000*st.mean(f[-1]['validation_position_m'] for f in fits),
                validation_epoch39_to49_change_mm=1000*st.mean(f[-1]['validation_position_m']-f[39]['validation_position_m'] for f in fits),
                validation_last10_minus_previous10_mm=1000*st.mean(st.mean(e['validation_position_m'] for e in f[40:50])-st.mean(e['validation_position_m'] for e in f[30:40]) for f in fits),
                validation_best_mean_mm=1000*st.mean(x['validation_score_m'] for x in selected)))
    sources = [M/'EVENTS.jsonl', M/'COST_AUDIT.json', M/'DEMO_SPLITS.json',
               R/'physical_effect_probe/readout_models.py', R/'physical_effect_probe/train_probe.py',
               R/'reference_source/libero/lifelong/models/modules/rgb_modules.py']
    result = dict(status='PASS_LOCAL_READONLY_CURVE_RECOMPUTATION',
        source_sha256={str(p.relative_to(R)):sha(p) for p in sources}, curves=curves,
        limitations=['Training SmoothL1 and validation position norms have different units; not a numerical train-test gap.',
                      'Selected checkpoint training-set positional metrics were not logged.',
                      'P3 bypasses the original frozen Temporal; it includes a separately trained light temporal readout.',
                      'P2 vs P3 also changes decoder capacity; not an isolated Temporal ablation.',
                      'No patch-token or matched no-temporal control exists yet.'],
        new_model_forwards=0, new_optimizer_steps=0, environment_episodes=0)
    (R/'BOTTLENECK_CURVE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps([x for x in curves if x['model']=='P3'],indent=2))

if __name__=='__main__':
    main()
