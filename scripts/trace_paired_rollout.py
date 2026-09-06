"""Passive vector-env observer around the unchanged official test/evaluator."""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import libero.lifelong.metric as metric
    target = Path(os.environ['ROLLOUT_TRACE_DIR'])
    mode = os.environ.get('ROLLOUT_TRACE_MODE', 'full')
    assert mode in ('full', 'light')
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / 'COMPLETE.json').exists(), 'Completed traces must not be overwritten'
    native_vector = metric.SubprocVectorEnv
    instances = []

    class ObservedVector(native_vector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.env_num == 10
            self.round = -1
            self.pending = False
            instances.append(self)

        def set_init_state(self, init_state, *args, **kwargs):
            self.flush()
            self.round += 1
            self.initial = np.array(init_state, copy=True)
            self.actions, self.dones, self.steps = [], [], []
            self.low, self.states, self.images, self.image_steps = {}, [], [], []
            self.step_number = -5
            self.pending = True
            obs = super().set_init_state(init_state, *args, **kwargs)
            if mode == 'full':
                self.observe(obs)
            return obs

        def observe(self, obs):
            keys = [k for k, v in obs[0].items() if np.asarray(v).size < 512
                    and np.issubdtype(np.asarray(v).dtype, np.number)]
            if not self.low:
                self.low = {k: [] for k in keys}
            assert set(keys) == set(self.low)
            for key in keys:
                self.low[key].append(np.stack([o[key] for o in obs]).copy())
            self.states.append(np.stack(super().get_sim_state()).copy())
            if self.step_number == 0 or (self.step_number > 0 and self.step_number % 50 == 0):
                self.images.append(np.stack([o['agentview_image'] for o in obs]).copy())
                self.image_steps.append(self.step_number)

        def step(self, action, *args, **kwargs):
            # Copies do not modify actions, observations, simulator, or any RNG.
            saved_action = np.array(action, copy=True)
            result = super().step(action, *args, **kwargs)
            self.step_number += 1
            self.actions.append(saved_action)
            self.dones.append(np.asarray(result[2]).copy())
            self.steps.append(self.step_number)
            if mode == 'full':
                self.observe(result[0])
            return result

        def flush(self):
            if not self.pending:
                return
            actions = np.stack(self.actions)
            done = np.stack(self.dones).astype(bool)
            steps = np.asarray(self.steps)
            values = dict(initial_states=self.initial, actions=actions, done=done, steps=steps)
            if mode == 'full':
                values.update({f'obs__{k}': np.stack(v) for k, v in self.low.items()})
                values.update(sim_states=np.stack(self.states),
                              observation_steps=np.concatenate(([-5], steps)),
                              agentview_images=np.stack(self.images), image_steps=np.asarray(self.image_steps))
            np.savez_compressed(target / f'round{self.round}.npz', **values)
            trials = []
            for lane in range(10):
                successes = steps[(steps > 0) & done[:, lane]]
                trials.append(dict(trial=self.round*10+lane, lane=lane, round=self.round,
                                   success=bool(len(successes)), first_success_step=int(successes[0]) if len(successes) else None,
                                   policy_steps=int(steps[-1])))
            (target / f'round{self.round}.json').write_text(json.dumps(trials, indent=2)+'\n')
            self.pending = False
            print(f'[trace] round={self.round} trials=10 successes={sum(r["success"] for r in trials)} mode={mode}', flush=True)

        def close(self):
            self.flush()
            return super().close()

    metric.SubprocVectorEnv = ObservedVector
    runpy.run_module('libero.lifelong.test', run_name='__main__')
    assert len(instances) == 1 and instances[0].round == 1
    assert all((target / f'round{i}.npz').is_file() for i in range(2))
    rows = sum([json.loads((target / f'round{i}.json').read_text()) for i in range(2)], [])
    (target / 'COMPLETE.json').write_text(json.dumps(dict(mode=mode, trials=20,
        successes=sum(r['success'] for r in rows), success_rate=sum(r['success'] for r in rows)/20,
        evaluator='unchanged libero/lifelong/test.py and metric.py; single requested task at fixed ep99',
        model_update_steps=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
