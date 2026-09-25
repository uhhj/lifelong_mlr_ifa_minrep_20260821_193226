import unittest
import numpy as np
from .common import TASKS, SEEDS, split_demos, balanced_windows
from .statistics import folds, predict_fold, reconstruct_targets, bootstrap


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(seed=seed, source=s, target=t, b_t=t, b_s=s,
                          S_visual=.01*s*t, S_language=.03*s, A=t-s,
                          S_lat=.02*s*t, S_GN=.01*(s+t), R_old=s/100, R_new=-t/100)
                     for seed in SEEDS for s in TASKS for t in TASKS if s != t]

    def test_task_excluded_in_both_roles(self):
        for f in folds(self.rows):
            self.assertEqual((len(f['train']), len(f['test'])), (18, 9))
            for i in f['train']:
                self.assertNotIn(f['held'], (self.rows[i]['source'], self.rows[i]['target']))

    def test_data_roles_disjoint_and_order_invariant(self):
        ds = ['demo%d' % i for i in range(50)]
        a = split_demos(6, ds)
        self.assertEqual(a, split_demos(6, list(reversed(ds))))
        self.assertEqual([len(a[k]) for k in ('A_train', 'D_signal', 'E_eval')], [30, 10, 10])
        self.assertEqual(len(set(sum(a.values(), []))), 50)

    def test_window_sampling_cap_and_unique(self):
        demos = ['d%d' % i for i in range(10)]
        rows = balanced_windows(demos, {d: 150 for d in demos}, 128)
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(set(rows)), 128)

    def test_heldout_features_do_not_fit_transform(self):
        f = folds(self.rows)[0]
        y = np.arange(36.)
        _, before = predict_fold(self.rows, 'M_LA', y, f['train'], f['test'])
        changed = [dict(r) for r in self.rows]
        for i in f['test']:
            changed[i]['S_lat'] += 1e10
        _, after = predict_fold(changed, 'M_LA', y, f['train'], f['test'])
        self.assertEqual(before, after)

    def test_constant_target_returns_training_mean(self):
        f = folds(self.rows)[0]
        pred, m = predict_fold(self.rows, 'M0', np.full(36, .25), f['train'], f['test'])
        self.assertTrue(np.all(pred == .25))
        self.assertTrue(m['constant_target'])

    def test_reference_arrays_shared(self):
        success = {}
        for seed in SEEDS:
            for t in TASKS:
                for model in ['base', 'reference%d' % t] + ['source%d' % s for s in TASKS]:
                    success[seed, model, t] = np.arange(20) % 2
                for s in TASKS:
                    if s != t:
                        for e in (s, t):
                            success[seed, 'transfer%d_%d' % (s, t), e] = np.arange(20) % 2
        indices = {(seed, t): np.arange(20)[::-1] for seed in SEEDS for t in TASKS}
        targets = reconstruct_targets(self.rows, success, indices)
        self.assertTrue(all(all(v == 0 for v in r.values()) for r in targets))
        boot = bootstrap(self.rows, success, count=2)
        self.assertTrue(all(v == 0 for r in boot for k, v in r.items() if k != 'repeat'))


if __name__ == '__main__':
    unittest.main()
