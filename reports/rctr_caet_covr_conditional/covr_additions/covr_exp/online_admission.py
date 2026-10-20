"""Bounded real native restoration check before any formal COVR training."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
from pathlib import Path
import numpy as np
import torch
from coreset_exp.common import O, SEED, write
from coreset_exp.model import cfg, data_setup, complete, strict_downstream
from libero.lifelong.metric import OffScreenRenderEnv
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed
from . import snapshot_runtime as rt


def run():
    assert not (O/"ONLINE_ADMISSION.json").exists()
    control_seed(SEED)
    torch.backends.cudnn.benchmark = True
    c = cfg("eval")
    bench, _, shape = data_setup(c, False)
    c.shape_meta = shape
    parent = Path(os.environ["RA_BASE"])
    checkpoint = parent/"ra_coreset/models/mlrifa_7_model_ep99.pth"
    policy = complete(c, shape)
    strict_downstream(policy, checkpoint)
    policy.eval().requires_grad_(False)
    task = bench.get_task(6)
    instruction, mask = bench.get_task_emb(6)
    initials = torch.load(Path(c.init_states_folder)/task.problem_folder/task.init_states_file)
    assert len(initials) >= 32
    rt.initialize(O/"preflight/online")
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
        camera_heights=c.data.img_h, camera_widths=c.data.img_w)
    try:
        env.seed(SEED)
        env.reset()
        env.seed(SEED)
        policy.reset()
        observation = env.set_init_state(initials[20])
        history = []
        for _ in range(5):
            observation, _ = rt.envstep(env, np.zeros(7), "engineering")
        noise = rt.noise(SEED*10000+7, 7)
        for step in range(7):
            history = rt.stack(history, observation)
            params, component = rt.params(
                policy, rt.batch_of(history, instruction, mask, c))
            observation, _ = rt.envstep(
                env, rt.normal(params, component, noise[step]), "engineering")
        snapshot = rt.snapshot(env, policy, observation, history[-7:])
        gate = rt.gate(env, policy, snapshot, 20, instruction, mask, c)
        observation, history, _ = rt.restore(env, policy, snapshot)
        data = rt.batch_of(rt.stack(history, observation), instruction, mask, c)
        with torch.no_grad():
            feature = policy.spatial_encode(policy.preprocess_input(data, train_mode=False))
            expected = policy.policy_head.forward_fn(policy.temporal_encode(feature)[:, -1])
        brief = BriefPolicy(cfg("ra_coreset"), None).cuda().eval()
        strict_downstream(brief, checkpoint)
        with torch.no_grad():
            actual = brief.policy_head.forward_fn(brief.temporal_encode(feature)[:, -1])
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        del brief
    finally:
        env.close()
    del policy
    torch.cuda.empty_cache()
    assert rt.COUNTS["engineering_steps"] == 28
    result = dict(status="PASS", engineering_steps=28, formal_episodes=0,
                  optimizer_updates=0, snapshot_gate=gate,
                  native_to_brief_gmm_exact=True, parent_checkpoint=str(checkpoint))
    write(O/"ONLINE_ADMISSION.json", result)
    return result
