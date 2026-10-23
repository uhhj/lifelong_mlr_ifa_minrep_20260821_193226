"""Frozen Temporal token reachability: CPU forward/input gradients, no optimization."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path
import torch
import transformers

torch.set_num_threads(1)
Q = Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
W = Q/'TACR_41'
S = W/'source'
entry = json.loads((W/'ENTRY.json').read_text())
O = Path(entry['output'])
root = O/'ra_coreset'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

contract = json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
assert not [p for p, h in contract['files'].items() if sha(p) != h]
cfg = json.loads((O/'resolved_ra_coreset.json').read_text())
assert all(cfg['policy'][k] is True for k in ('use_language', 'use_vision', 'use_state'))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

torch.manual_seed(932741)
tm = load_module('token_probe_transformer', S/'libero/lifelong/models/modules/transformer_modules.py')
sm = load_module('token_probe_student', S/'caet_v3_exp/model.py')
model = tm.OfficialGPT2Decoder(embed_size=512, num_layers=6, num_heads=8,
                              dropout=.15, max_position_embeddings=32, train=True).cpu()
cp = root/'models/mlrifa_9_model_ep99.pth'
raw = torch.load(cp, map_location='cpu')
weights = {k.split('temporal_transformer.', 1)[1]: v for k, v in raw['state_dict'].items() if 'temporal_transformer.' in k}
model.load_state_dict(weights, strict=True)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
student_path = root/'aux_heads/task9.pth'
student = sm.Student().cpu()
student.load_state_dict(torch.load(student_path, map_location='cpu')['model'], strict=True)
student.eval()
for p in student.parameters():
    p.requires_grad_(False)

masks = [b.attn.bias[0, 0, :32, :32].bool() for b in model.decoder.h]
assert len(masks) == 6
assert all(torch.equal(m, torch.tril(torch.ones(32, 32, dtype=torch.bool))) for m in masks)
assert masks[0][28].nonzero().flatten().tolist() == list(range(29))
x = torch.randn(1, 8, 4, 512)
direction = .2*torch.sin(torch.arange(512, dtype=torch.float32)*.13)
action = torch.zeros(1, 7)
forwards = 0

def encode(value):
    global forwards
    forwards += 1
    b, t, m, e = value.shape
    return model(value.reshape(b, t*m, e))

def effect_readout(h):
    c = student.deploy.R(h)
    return c, student.D(student.condition(c, action))

with torch.no_grad():
    base = encode(x)
    base_h = base[:, 28]
    base_c, base_effect = effect_readout(base_h)
    rows = []
    for idx in [0, 3, 25, 26, 27, 28, 29, 30, 31]:
        changed = x.clone().reshape(1, 32, 512)
        changed[:, idx] += direction
        y = encode(changed.reshape_as(x))
        c, effect = effect_readout(y[:, 28])
        row = dict(input_flat_index=idx, time_index=idx//4, modality_index=idx%4,
                   selected28_max_abs_change=float((y[:, 28]-base_h).abs().max()),
                   full_last31_max_abs_change=float((y[:, 31]-base[:, 31]).abs().max()),
                   learned_R_max_abs_change=float((c-base_c).abs().max()),
                   learned_effect_fixed_action_max_abs_change=float((effect-base_effect).abs().max()))
        if idx > 28:
            assert row['selected28_max_abs_change'] == row['learned_R_max_abs_change'] == row['learned_effect_fixed_action_max_abs_change'] == 0
            assert row['full_last31_max_abs_change'] > 0
        if idx in (25, 26, 27):
            assert row['selected28_max_abs_change'] > 0
        rows.append(row)
    joint = x.clone()
    joint[:, -1, 1:] += direction
    joint_y = encode(joint)
    assert torch.equal(joint_y[:, 28], base_h)
    first = x[:, :1].clone()
    first_y = encode(first)
    first[:, 0, 1:] += direction
    first_changed = encode(first)
    first_invisible = float((first_changed[:, 0]-first_y[:, 0]).abs().max())
    assert first_invisible == 0

gx = x.clone().requires_grad_(True)
gout = encode(gx)
readout = torch.cos(torch.arange(512, dtype=torch.float32)*.07)
input_grad = torch.autograd.grad((gout[:, 28]*readout).sum(), gx)[0]
grad_norms = input_grad.reshape(32, 512).norm(dim=-1)
assert torch.equal(grad_norms[29:], torch.zeros(3))
assert bool((grad_norms[[25, 26, 27, 28]] > 0).all())
assert all(p.grad is None for p in model.parameters())
assert all(p.grad is None for p in student.parameters())
assert not torch.cuda.is_initialized()
assert not [p for p, h in contract['files'].items() if sha(p) != h]
source_paths = dict(brief=S/'libero/lifelong/models/brief.py',
                    complete=S/'libero/lifelong/models/complete_policy.py',
                    transformer=S/'libero/lifelong/models/modules/transformer_modules.py',
                    physical_branch=S/'caet_v3_exp/model.py',
                    observer=S/'caet_v3_exp/observe.py',
                    teacher=S/'caet_v3_exp/teacher_keys.py',
                    native_rollout=S/'coreset_exp/eval_metric.py')
print(json.dumps(dict(status='FROZEN_TEMPORAL_MODALITY_VISIBILITY_PROVEN_ZERO_TRAINING',
    observed_unix=time.time(), variant='TACR', seed=41, checkpoint_stage=9,
    input_kind='deterministic_synthetic_latent_no_physical_performance_claim',
    input_shape=[1, 8, 4, 512], readout_flat_index=28, full_last_index=31,
    selected_visible_input_indices=list(range(29)), selected_invisible_input_indices=[29, 30, 31],
    slot_semantics=['language', 'language_conditioned_agentview', 'language_conditioned_eye_in_hand', 'language_conditioned_joint_and_gripper'],
    config_policy_flags={k:cfg['policy'][k] for k in ('use_language', 'use_vision', 'use_state')},
    config_obs_modalities=cfg['data']['obs']['modality'],
    all_six_layers_standard_causal_mask=True,
    selected_ignores_joint_current_visual_state_perturbation=True,
    one_step_selected_ignores_visual_state_max_abs_change=first_invisible,
    perturbations=rows, input_gradient_l2_by_time_modality=grad_norms.reshape(8,4).tolist(),
    model_forwards=forwards, input_gradient_calls=1, parameter_gradients_created=False,
    optimizer_updates=0, environment_steps=0, cuda_initialized=False, server_writes=0,
    physical_generalization_or_effect_quality_measured=False,
    checkpoint_sha256=sha(cp), student_sha256=sha(student_path),
    loaded_temporal_keys=len(weights), frozen_files=len(contract['files']),
    frozen_before_after_unchanged=True,
    sources={k:dict(path=str(p),sha256=sha(p)) for k,p in source_paths.items()},
    transformer_version=transformers.__version__,torch_version=torch.__version__,
    actual_attention_source=inspect.getsource(type(model.decoder.h[0].attn)._attn))))
