import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("rc_core", ROOT / "src/replay_constraint.py")
from summarize import read_fixed_matrix, score_matrix, diagnostic_summary
from run import commands


class TinyPolicy(torch.nn.Module):
    """Stochastic forward and distribution NLL, including an unused parameter."""
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.temporal_transformer = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.25))
        self.policy_head = torch.nn.Linear(6, 10)
        self.policy_head.unused = torch.nn.Parameter(torch.tensor([.5]))

    def compute_loss(self, data, task, index, reduction="mean"):
        z = self.temporal_transformer(data["feature"])
        out = self.policy_head(z)
        means = out[:, :4].reshape(-1, 2, 2).tanh()
        scales = torch.nn.functional.softplus(out[:, 4:8].reshape(-1, 2, 2)) + 1e-4
        mixture = torch.distributions.Categorical(logits=out[:, 8:])
        components = torch.distributions.Independent(torch.distributions.Normal(means, scales), 1)
        distribution = torch.distributions.MixtureSameFamily(mixture, components)
        losses = -distribution.log_prob(data["actions"])
        return (losses.mean() if reduction == "mean" else losses), z


class ProjectionTests(unittest.TestCase):
    def project(self, g, r, **kwargs):
        p = torch.nn.Parameter(torch.zeros(len(g), dtype=torch.float64))
        p.grad = torch.tensor(g, dtype=torch.float64)
        before = p.grad.clone()
        report = core.project_parameter_grads_([p], [torch.tensor(r, dtype=torch.float64)], **kwargs)
        return p.grad, report, before

    def test_conflict_exact_projection(self):
        grad, m, _ = self.project([-2, 3], [1, 0])
        torch.testing.assert_close(grad, torch.tensor([0., 3.], dtype=torch.float64))
        self.assertTrue(m["projected"])
        self.assertAlmostEqual(m["dot_after"], 0)

    def test_aligned_and_orthogonal_unchanged(self):
        for g in ([2, 3], [0, 3]):
            grad, m, before = self.project(g, [1, 0])
            self.assertTrue(torch.equal(grad, before))
            self.assertFalse(m["projected"])

    def test_zero_tiny_and_disabled_reference(self):
        for r in ([0, 0], [1e-15, 0]):
            grad, m, before = self.project([-1, 2], r)
            self.assertTrue(torch.equal(grad, before))
            self.assertTrue(m["tiny_reference"])
        grad, m, before = self.project([-1, 2], [1, 0], enabled=False)
        self.assertTrue(torch.equal(grad, before))
        self.assertTrue(m["conflict"])
        self.assertFalse(m["projected"])

    def test_global_not_per_block(self):
        p, q = (torch.nn.Parameter(torch.zeros(1)) for _ in range(2))
        p.grad, q.grad = torch.tensor([-1.]), torch.tensor([2.])
        m = core.project_parameter_grads_([p, q], [torch.ones(1), torch.ones(1)])
        self.assertFalse(m["projected"])
        self.assertEqual(p.grad.item(), -1.)

    def test_total_not_new_projection_and_no_double_replay(self):
        # g_new=-3, g_old=2, original batch weights new=.2, old=.8.
        # Mixed gradient=1 is aligned, even though new vs old is conflicting.
        p = torch.nn.Parameter(torch.tensor(0.))
        losses = torch.stack([-3*p, 2*p, 2*p, 2*p, 2*p])
        loss, _, old, n_new, n_old = core.mixed_losses(losses, [7, 6, 6, 6, 6], 7)
        m = core.backward_mixed_with_reference(loss, old, [p])
        self.assertAlmostEqual(p.grad.item(), 1., places=6)
        self.assertEqual((n_new, n_old), (1, 4))
        self.assertFalse(m["projected"])

    def test_new_only_and_old_only(self):
        for ids in ([7, 7], [6, 6]):
            p = torch.nn.Parameter(torch.tensor(1.))
            losses = torch.stack([p.square(), 2*p.square()])
            loss, _, old, _, _ = core.mixed_losses(losses, ids, 7)
            m = core.backward_mixed_with_reference(loss, old, [p], loss_scale=2)
            self.assertEqual(p.grad.item(), 6.)
            self.assertFalse(m["projected"])

    def test_nonfinite_aborts_before_mutation(self):
        with self.assertRaises(FloatingPointError):
            self.project([float("nan"), 1], [1, 1])

    def test_invalid_task_and_reduction(self):
        with self.assertRaises(ValueError):
            core.mixed_losses(torch.ones(2), [6, 8], 7)
        with self.assertRaises(ValueError):
            core.mixed_losses(torch.ones(2, 1), [6, 7], 7)

    def test_loss_scale_and_clipping_keep_projected_halfspace(self):
        p = torch.nn.Parameter(torch.tensor([1., 1.]))
        losses = torch.stack([p[0] * 2, -p[0] * 6 + p[1] * 8])
        loss, _, old, _, _ = core.mixed_losses(losses, [6, 7], 7)
        m = core.backward_mixed_with_reference(loss, old, [p], loss_scale=3)
        torch.nn.utils.clip_grad_norm_([p], .5)
        self.assertTrue(m["projected"])
        self.assertGreaterEqual(float(p.grad[0]), -1e-6)
        self.assertAlmostEqual(float(p.grad.norm()), .5, places=5)

    def test_random_projection_minimality_and_idempotence(self):
        generator = torch.Generator().manual_seed(9)
        for _ in range(20):
            g, r = torch.randn(20, generator=generator, dtype=torch.float64), torch.randn(20, generator=generator, dtype=torch.float64)
            g -= (abs(g.dot(r)) + 1) * r / r.dot(r)
            out, m, _ = self.project(g.tolist(), r.tolist())
            expected = g - min(float(g.dot(r)), 0.) / r.dot(r) * r
            torch.testing.assert_close(out, expected)
            self.assertGreaterEqual(float(out.dot(r)), -1e-12)
            twice, _, _ = self.project(out.tolist(), r.tolist())
            torch.testing.assert_close(twice, out, atol=1e-12, rtol=1e-12)


class ParityTests(unittest.TestCase):
    def test_official_gmm_and_base_loss_reduction(self):
        # Read the actual supplied policy code without importing its simulator stack.
        model_dir = ROOT.parent / "publish_repo/libero/lifelong/models"
        if not model_dir.is_dir():
            self.skipTest("Optional local source snapshot absent; run parity_real.py on the server.")
        tree = ast.parse((model_dir / "policy_head.py").read_text(encoding="utf-8"))
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GMMHead")
        namespace = {"torch": torch, "nn": torch.nn, "D": torch.distributions,
                     "F": torch.nn.functional}
        exec(compile(ast.Module(body=[head], type_ignores=[]), "actual_GMMHead", "exec"), namespace)
        tree = ast.parse((model_dir / "base_policy.py").read_text(encoding="utf-8"))
        base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BasePolicy")
        compute = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
        exec(compile(ast.Module(body=[compute], type_ignores=[]), "actual_compute_loss", "exec"), namespace)
        class Policy(torch.nn.Module):
            compute_loss = namespace["compute_loss"]
            def __init__(self):
                super().__init__()
                self.temporal_transformer = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.15))
                self.policy_head = namespace["GMMHead"](6, 2, hidden_size=8, num_layers=1, num_modes=2)
            def forward(self, x):
                z = self.temporal_transformer(x[:, -1, 0, :])
                return self.policy_head(z), z
        torch.manual_seed(19)
        model = Policy()
        data = {"feature": torch.randn(5, 1, 3, 1, 4), "actions": torch.randn(5, 1, 3, 2)}
        state = torch.get_rng_state()
        loss, _ = model.compute_loss(data, None, None)
        expected = torch.autograd.grad(loss, tuple(model.parameters()))
        torch.set_rng_state(state)
        per_sample, _ = model.compute_loss(data, None, None, reduction="none")
        mix, _, old, _, _ = core.mixed_losses(per_sample, [6, 7, 6, 7, 7], 7)
        self.assertTrue(torch.equal(loss, mix))
        core.backward_mixed_with_reference(mix, old, model.parameters(), enabled=False)
        for grad, parameter in zip(expected, model.parameters()):
            self.assertTrue(torch.equal(grad, parameter.grad))

    def test_shadow_matches_native_gmm_adamw_and_rng(self):
        torch.manual_seed(42)
        native = TinyPolicy()
        shadow = copy.deepcopy(native)
        opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=.1) for m in (native, shadow)]
        scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda step: min((step + 1)/4, 1.)) for o in opts]
        for ids in ([7]*6, [6, 7, 6, 7, 6, 7], [6]*6, [6, 6, 6, 7, 7, 7]):
            data = {"feature": torch.randn(6, 4), "actions": torch.randn(6, 2)}
            state = torch.get_rng_state()
            opts[0].zero_grad()
            loss, _ = native.compute_loss(data, None, None)
            (loss * 1.7).backward()
            torch.nn.utils.clip_grad_norm_(native.parameters(), .4)
            opts[0].step(); scheds[0].step()
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(state)
            opts[1].zero_grad()
            individual, _ = shadow.compute_loss(data, None, None, reduction="none")
            mixed, _, old, _, _ = core.mixed_losses(individual, ids, 7)
            core.backward_mixed_with_reference(mixed, old, [p for p in shadow.parameters() if p.requires_grad], loss_scale=1.7, enabled=False)
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), .4)
            opts[1].step(); scheds[1].step()
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for a, b in zip(native.parameters(), shadow.parameters()):
                self.assertTrue(torch.equal(a, b))
            for i in opts[0].state_dict()["state"]:
                for key, val in opts[0].state_dict()["state"][i].items():
                    self.assertTrue(torch.equal(val, opts[1].state_dict()["state"][i][key]))
            self.assertEqual(scheds[0].state_dict(), scheds[1].state_dict())
            self.assertIsNone(shadow.policy_head.unused.grad)

    def test_adapter_observe_and_epoch_logs(self):
        # Exercise the actual subclass observe method without simulator imports.
        fake = ModuleType("libero.lifelong.algos.mlr")
        class MLR(torch.nn.Module):
            def __init__(self, n_tasks, cfg, **kwargs):
                super().__init__()
                self.cfg, self.experiment_dir, self.loss_scale = cfg, cfg.experiment_dir, cfg.train.loss_scale
                self.policy = TinyPolicy()
                self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-3)
                self.steps = 0
            def map_tensor_to_device(self, data):
                return data
            def optimizer_step_and_maybe_schedule(self):
                self.optimizer.step()
                self.steps += 1
        fake.MLR = MLR
        keys = {"libero.lifelong.algos.mlr": fake, "libero.lifelong.algos.replay_constraint": core}
        old_modules = {key: sys.modules.get(key) for key in keys}
        sys.modules.update(keys)
        try:
            mod = load("rc_adapter", ROOT / "src/mlr_replay_constraint.py")
            with tempfile.TemporaryDirectory() as directory:
                cfg = NS(ifa_enabled=False, experiment_dir=directory, seed=42,
                         train=NS(loss_scale=1., grad_clip=100.),
                         lifelong=NS(projection_enabled=True, reference_norm_floor=1e-12,
                                     synchronize_timing=True, step_log_every=1))
                algo = mod.MLRReplayConstraint(1, cfg)
                for epoch in (0, 0, 1):
                    algo.observe([{"feature": torch.randn(4, 4), "actions": torch.randn(4, 2)}, ["6", "7", "6", "7"]], 7, epoch)
                algo._flush_epoch()
                self.assertEqual(algo.steps, 3)
                summary = diagnostic_summary(Path(directory) / "replay_constraint_epochs.csv")
                self.assertEqual(summary["batches"], 3)
                self.assertEqual(summary["epoch_rows"], 2)
                self.assertFalse(summary["complete_epoch_grid"])
                self.assertEqual(len((Path(directory) / "replay_constraint_steps.jsonl").read_text().splitlines()), 3)
        finally:
            for key, value in old_modules.items():
                if value is None:
                    del sys.modules[key]
                else:
                    sys.modules[key] = value


class ProtocolTests(unittest.TestCase):
    def test_installer_is_additive_and_idempotent(self):
        from install import plan, IMPORT
        snippets = {
            "libero/lifelong/algos/mlr.py": "class MLR: pass\n# build_dataset_for_task save_final_checkpoint\n",
            "libero/lifelong/algos/base.py": "def observe(): pass\n# optimizer_step_and_maybe_schedule batch_step\n",
            "libero/lifelong/models/base_policy.py": "def compute_loss(reduction=None): pass\n",
            "libero/lifelong/models/brief.py": "class BriefPolicy: pass\n# temporal_transformer policy_head\n",
            "libero/lifelong/models/policy_head.py": '# reduction == "none"\n',
            "libero/lifelong/algos/__init__.py": "# get_algo_class\n",
            "libero/lifelong/test.py": "# eval_epochs\n",
        }
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            for name, text in snippets.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            changes = plan(repo)
            self.assertEqual(len(changes), 4)
            for path, content in changes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            self.assertEqual(plan(repo), [])
            self.assertEqual((repo / "libero/lifelong/algos/base.py").read_text(), snippets["libero/lifelong/algos/base.py"])
            self.assertIn(IMPORT, (repo / "libero/lifelong/algos/__init__.py").read_text())
            (repo / "libero/lifelong/algos/replay_constraint.py").write_text("# different user file")
            with self.assertRaises(ValueError):
                plan(repo)

    def test_fixed_epoch_metrics_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eval.log"
            lines = ["[task %d epoch 99 and %d succ.] 0.8" % (t, s) for s in range(6, 10) for t in range(6, s+1)]
            path.write_text("\n".join(lines))
            result = score_matrix(read_fixed_matrix(path))
            self.assertEqual((result["fwt"], result["nbt"], result["auc"]), (80., 0., 80.))
            for bad in (lines[:-1], lines + lines[:1], [x.replace("epoch 99", "epoch 95") for x in lines]):
                path.write_text("\n".join(bad))
                with self.assertRaises(ValueError):
                    read_fixed_matrix(path)

    def test_launcher_protocol(self):
        args = NS(repo=Path("/example"), tag="test", smoke=False, seed=42,
                  feature_root=None, data_root=None, checkpoint=None, python="python",
                  action="train", arm="constraint", scheduler_mode="batch_step")
        cmd = commands(args)
        self.assertIn("+save_final_checkpoint=true", cmd)
        self.assertIn("+ifa_enabled=false", cmd)
        self.assertIn("train.batch_size=10", cmd)
        self.assertIn("lifelong.projection_enabled=true", cmd)
        args.action = "eval"
        cmd = commands(args)
        self.assertIn("+eval_epochs=[99]", cmd)
        self.assertNotIn("+eval_current_only=true", cmd)
        self.assertIn("eval.n_eval=20", cmd)
        args.action, args.smoke = "train", True
        self.assertIn("+end_task=8", commands(args))
        self.assertIn("+max_optimizer_updates=4", commands(args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
