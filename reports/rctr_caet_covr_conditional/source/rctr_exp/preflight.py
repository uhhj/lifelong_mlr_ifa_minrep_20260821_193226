"""Real CUDA/native restoration admission. Zero optimizer updates, no evaluation."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O, SEED, verify, write, rng, restore_rng
from coreset_exp.model import cfg, data_setup, complete, strict_downstream
from coreset_exp.train import PaperMLR, build
from libero.lifelong.metric import OffScreenRenderEnv
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed
from .observe import RCTRMLR
from .loss import objective
from . import snapshot_runtime as rt


def grads(model):
    return {n: p.grad.detach().clone() if p.grad is not None else None
            for n, p in model.named_parameters()}


def main():
    verify()
    assert not (O/"GPU_PREFLIGHT.json").exists(), "Do not repeat a passed admission"
    assert torch.cuda.is_available()
    control_seed(SEED)
    torch.backends.cudnn.benchmark = True
    ec = cfg("eval")
    bench, _, shape = data_setup(ec, False)
    ec.shape_meta = shape
    policy = complete(ec, shape)
    parent = Path(os.environ["RA_BASE"])
    strict_downstream(policy, parent/"ra_coreset/models/mlrifa_6_model_ep99.pth")
    policy.eval().requires_grad_(False)
    task = bench.get_task(6)
    instr, mask = bench.get_task_emb(6)
    initials = torch.load(Path(ec.init_states_folder)/task.problem_folder/task.init_states_file)
    assert len(initials) >= 26
    rt.initialize(O/"preflight")
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(ec.bddl_folder)/task.problem_folder/task.bddl_file),
        camera_heights=ec.data.img_h, camera_widths=ec.data.img_w)
    try:
        env.seed(SEED)
        env.reset()
        env.seed(SEED)
        policy.reset()
        obs = env.set_init_state(initials[20])
        history = []
        for _ in range(5):
            obs, _ = rt.envstep(env, np.zeros(7), "engineering")
        noise = rt.noise(SEED*10000+7, 7)
        for step in range(7):
            history = rt.stack(history, obs)
            par, k = rt.params(policy, rt.batch_of(history, instr, mask, ec))
            obs, _ = rt.envstep(env, rt.normal(par, k, noise[step]), "engineering")
        snap = rt.snapshot(env, policy, obs, history[-7:])
        gate = rt.gate(env, policy, snap, 20, instr, mask, ec)
        obs, hist, _ = rt.restore(env, policy, snap)
        data = rt.batch_of(rt.stack(hist, obs), instr, mask, ec)
        with torch.no_grad():
            feature = policy.spatial_encode(policy.preprocess_input(data, train_mode=False))
            expected = policy.policy_head.forward_fn(policy.temporal_encode(feature)[:, -1])
        bc = cfg("ra_coreset")
        brief = BriefPolicy(bc, None).cuda().eval()
        strict_downstream(brief, parent/"ra_coreset/models/mlrifa_6_model_ep99.pth")
        with torch.no_grad():
            actual = brief.policy_head.forward_fn(brief.temporal_encode(feature)[:, -1])
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        del brief
    finally:
        env.close()
    del policy
    torch.cuda.empty_cache()
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    try:
        bc = cfg("ra_coreset")
        bc.run_root = str(O/"preflight/smoke")
        bc.experiment_dir = str(O/"preflight/smoke/models")
        bc.buffer_dir = str(O/"preflight/smoke/buffer")
        a, _ = build(bc)
        paths = sorted((O/"features/6").glob("*.pt"))[:10]
        payload = [(torch.load(p, map_location="cpu"), "6") for p in paths]
        loader = DataLoader(payload, batch_size=10, num_workers=0)
        a.start_task(-1, loader)
        a.refs = {6: payload[0][0]["feature"][0, 0, 0].cuda()}
        a.rctr_records = []
        a.rctr_epoch_statistics = []
        a.rctr_scale = torch.load(O/"pretraining_action_scale.pth").cuda()
        # Stub only the isolated probe's final update and scheduler call. Neither
        # policy parameters nor optimizer state is updated in this admission.
        a.optimizer.step = lambda *args, **kwargs: None
        a.scheduler.step = lambda *args, **kwargs: None
        batch = next(iter(loader))
        state = rng()
        before = {k: v.detach().clone() for k, v in a.policy.state_dict().items()}
        native = PaperMLR.observe(a, copy.deepcopy(batch), 6, 0, {"selected_pairs": []})
        g1 = grads(a.policy)
        restore_rng(state)
        a.step = 0
        augmented = RCTRMLR.observe(a, copy.deepcopy(batch), 6, 0, {"selected_pairs": []})
        g2 = grads(a.policy)
        assert native == augmented
        assert all((g1[k] is None and g2[k] is None) or
                   (g1[k] is not None and g2[k] is not None and torch.equal(g1[k], g2[k]))
                   for k in g1)
        assert all(torch.equal(v, before[k]) for k, v in a.policy.state_dict().items())
        # Synthetic labels test numerical/differentiation mechanics ONLY.
        sample = payload[0][0]
        record = dict(zdelta=sample["feature"][0], z0=sample["feature"][0],
            a_plus=sample["actions"][0, -1],
            a_minus=-sample["actions"][0, -1],
            a0=sample["actions"][0, -1], has_normal_reference=True)
        a.optimizer.zero_grad()
        before_rng = rng()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1234567)
            loss, parts = objective(a.policy.module, record, a.rctr_scale, torch.device("cuda"))
            loss.backward()
        assert torch.equal(before_rng["cpu"], torch.get_rng_state())
        assert all(torch.equal(x, y) for x, y in zip(before_rng["cuda"], torch.cuda.get_rng_state_all()))
        gs = [p.grad for p in a.policy.parameters() if p.grad is not None]
        assert gs and all(torch.isfinite(g).all() for g in gs)
        assert sum(float(g.abs().sum()) for g in gs) > 0
    finally:
        dist.destroy_process_group()
    verify()
    write(O/"GPU_PREFLIGHT.json", dict(status="PASS", optimizer_updates=0,
        formal_episodes=0, engineering_steps=rt.COUNTS["engineering_steps"],
        complete_snapshot_gate=gate, native_to_brief_gmm_exact=True,
        zero_record_loss_gradient_exact=True, auxiliary_rng_restored=True,
        auxiliary_synthetic_gradient_finite=True,
        synthetic_labels_are_not_experiment_evidence=True,
        certified_record_gradient_check="Required at first real sidecar before use",
        torch=torch.__version__, gpu=torch.cuda.get_device_name()))


if __name__ == "__main__":
    main()
