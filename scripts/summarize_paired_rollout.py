"""Summarize all paired trials with fixed geometric descriptions, not causal labels."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/paired_rollout_diagnosis'
OUT = ROOT / 'reports/paired_rollout_diagnosis'
CASES = ((43, 7, 7), (42, 9, 7), (43, 9, 9))
ARMS = ('mlr', 'release_ifa', 'corrected_ifa')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def first_step(steps, condition):
    matches = steps[condition]
    return int(matches[0]) if len(matches) else None


def main():
    assert json.loads((OUT / 'INSTRUMENTATION_PARITY.json').read_text())['verdict'] == 'PASS'
    historical = json.loads((ROOT / 'reports/own_reference_control/CONTROLLED_RESULTS.json').read_text())['results']
    rows, summaries, pairs = [], [], []
    traces = {}
    for seed, stage, task in CASES:
        case = f'seed{seed}_stage{stage}_task{task}'
        object_key = 'milk_1_pos' if task == 7 else 'orange_juice_1_pos'
        for arm in ARMS:
            path = RUN / case / arm
            complete = json.loads((path / 'COMPLETE.json').read_text())
            assert complete['trials'] == 20 and complete['mode'] == 'full'
            official = re.findall(r'\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)', (path / 'success.log').read_text())
            assert len(official) == 1 and tuple(map(int, official[0][:3])) == (task, 99, stage)
            assert abs(float(official[0][3])-complete['success_rate']) < 1e-9
            arm_rows = []
            for round_id in range(2):
                with np.load(path / f'round{round_id}.npz') as trace:
                    initial = trace['initial_states']
                    if arm == 'mlr':
                        traces[case, round_id] = initial.copy()
                    else:
                        assert np.array_equal(traces[case, round_id], initial), 'Unpaired initial states'
                    obs_steps = trace['observation_steps']
                    action_steps = trace['steps']
                    done = trace['done']
                    assert (action_steps[0], obs_steps[0]) == (-4, -5)
                    assert np.isfinite(trace['actions']).all() and np.isfinite(trace['sim_states']).all()
                    for lane in range(10):
                        success_step = first_step(action_steps, (action_steps > 0) & done[:, lane])
                        end = success_step if success_step is not None else int(action_steps[-1])
                        keep = (obs_steps >= 0) & (obs_steps <= end)
                        steps = obs_steps[keep]
                        obj = trace[f'obs__{object_key}'][keep, lane]
                        basket = trace['obs__basket_1_pos'][keep, lane]
                        eef = trace['obs__robot0_eef_pos'][keep, lane]
                        height = obj[:, 2] - obj[0, 2]
                        xy = np.linalg.norm(obj[:, :2] - basket[:, :2], axis=1)
                        distance = np.linalg.norm(obj - eef, axis=1)
                        lift = first_step(steps, height >= .03)
                        near = first_step(steps, xy <= .10)
                        category = ('success' if success_step is not None else 'no_3cm_rise' if lift is None
                                    else 'rise_but_not_near_basket' if near is None else 'near_basket_no_success')
                        row = dict(case=case, seed=seed, stage=stage, task=task, arm=arm,
                                   trial=round_id*10+lane, success=success_step is not None,
                                   first_success_step=success_step, analyzed_until_step=end,
                                   first_3cm_rise_step=lift, first_basket_xy_10cm_step=near,
                                   max_height_rise_m=float(height.max()), min_basket_xy_m=float(xy.min()),
                                   min_eef_target_m=float(distance.min()), failure_description=category)
                        arm_rows.append(row)
                        traces[case, arm, row['trial']] = dict(row=row, object=obj, steps=steps,
                            settled_state=trace['sim_states'][np.flatnonzero(obs_steps == 0)[0], lane].copy())
            assert len(arm_rows) == 20 and sum(r['success'] for r in arm_rows) == complete['successes']
            rows.extend(arm_rows)
            summaries.append(dict(case=case, arm=arm, trials=20, successes=complete['successes'],
                                  success_pct=100*complete['success_rate'],
                                  historical_success_pct=historical[arm]['per_seed'][str(seed)]['r'][stage-6][task-6],
                                  descriptions=dict(Counter(r['failure_description'] for r in arm_rows))))
        for control in ('mlr', 'release_ifa'):
            for trial in range(20):
                new = traces[case, 'corrected_ifa', trial]
                old = traces[case, control, trial]
                assert np.array_equal(new['settled_state'], old['settled_state']), 'Settled state differs before policy acts'
                n = min(len(new['steps']), len(old['steps']))
                assert np.array_equal(new['steps'][:n], old['steps'][:n])
                distance = np.linalg.norm(new['object'][:n]-old['object'][:n], axis=1)
                pairs.append(dict(case=case, control=control, trial=trial,
                                  corrected_success=new['row']['success'], control_success=old['row']['success'],
                                  corrected_description=new['row']['failure_description'],
                                  control_description=old['row']['failure_description'],
                                  max_target_separation_before_either_ends_m=float(distance.max())))
    assert len(rows) == 180 and len(pairs) == 120
    write_csv('per_trial.csv', rows)
    write_csv('paired_trials.csv', pairs)
    paired_summary = []
    for case in [f'seed{s}_stage{st}_task{t}' for s, st, t in CASES]:
        for control in ('mlr', 'release_ifa'):
            subset = [r for r in pairs if r['case'] == case and r['control'] == control]
            paired_summary.append(dict(case=case, control=control,
                both_success=sum(r['corrected_success'] and r['control_success'] for r in subset),
                corrected_only=sum(r['corrected_success'] and not r['control_success'] for r in subset),
                control_only=sum(not r['corrected_success'] and r['control_success'] for r in subset),
                both_failure=sum(not r['corrected_success'] and not r['control_success'] for r in subset)))
    result = dict(verdict='PAIRED_ROLLOUTS_COMPLETE', trials=180, instrumentation_control_trials=20,
                  paired_initial_and_settled_states=True, summaries=summaries, paired_summary=paired_summary,
                  scope='Selected-task diagnostic invocation; not replacement historical metrics. Geometric labels are not grasp or causal labels; no independent action-noise replicate.')
    (OUT / 'SUMMARY.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
