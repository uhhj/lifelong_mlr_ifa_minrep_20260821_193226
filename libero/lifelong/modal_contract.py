import hashlib
import json
from pathlib import Path
import torch
from libero.lifelong.runbook_core import sha256, strip_prefix, atomic_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def validate_flags(cfg):
    for key in ('use_vision', 'use_language', 'use_state'):
        if cfg.policy.get(key) is not True:
            raise ValueError('Expected boolean True: policy.' + key)


def tensor_state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def frontend_hash(state):
    state = strip_prefix(state)
    return tensor_state_hash({k: v for k, v in state.items()
                              if not k.startswith(('temporal_transformer.', 'policy_head.'))})


def build_contract(cfg, patch_path):
    validate_flags(cfg)
    checkpoint = torch.load(cfg.pretrain, map_location='cpu')
    state = strip_prefix(checkpoint['state_dict'])
    required = {'state_encoder.0.weight': (512, 9), 'state_encoder.0.bias': (512,),
                'state_encoder.2.weight': (512, 512), 'state_encoder.2.bias': (512,)}
    for prefix in ('state_scale', 'state_shift'):
        for layer in (0, 2):
            required[prefix + '.%d.weight' % layer] = (512, 512)
            required[prefix + '.%d.bias' % layer] = (512,)
    for key, shape in required.items():
        assert key in state and tuple(state[key].shape) == shape, (key, shape)
    body = {'schema_version': 2, 'use_vision': True, 'use_language': True, 'use_state': True,
        'token_order': ['language', 'agentview', 'eye_in_hand', 'state'],
        'state_keys': ['joint_states', 'gripper_states'], 'state_dims': [7, 2],
        'window_length': 8, 'embedding_dim': 512,
        'official_checkpoint_sha256': sha256(cfg.pretrain),
        'frozen_frontend_sha256': frontend_hash(state),
        'pretrain_use_state': checkpoint['cfg']['policy'].get('use_state', 'unknown'),
        'effective_use_state': True, 'pretrain_metadata_gap_status': 'DISCLOSED_GAP',
        'code_commit': 'a721807f2f7fa7f9833dcc85c592d78e9accd00b',
        'code_patch_sha256': sha256(patch_path),
        'export': {'batch_size': 1, 'module_mode': 'train', 'cfg_training': False,
            'augmentation': True, 'use_augmentation': True,
            'obs_preprocessing': 'robomimic process_obs; RGB uint8 to CHW float [0,1]; source CLIP normalization',
            'augmentation_layers': ['BatchWiseImgColorJitterAug', 'RandomAffineAug', 'RandomErasingAug'],
            'augmentation_config': json.loads(json.dumps({k: cfg.policy[k] for k in ['color_aug', 'translation_aug', 'erase_aug']})),
            'sampler': 'RandomSampler without replacement', 'seed': 42}}
    return dict(body, fingerprint=fingerprint(body))


def validate_contract(received, expected):
    for value in (received, expected):
        if not isinstance(value, dict) or 'fingerprint' not in value:
            raise ValueError('Missing verifiable modal contract; old caches forbidden')
        body = {k: v for k, v in value.items() if k != 'fingerprint'}
        if fingerprint(body) != value['fingerprint']:
            raise ValueError('Modal fingerprint mismatch')
    if received != expected:
        raise ValueError('Modal contract content mismatch; cross-modal resume forbidden')


def check_metadata(meta, expected):
    validate_contract(meta.get('modal_contract'), expected)
    if meta.get('modal_fingerprint') != expected['fingerprint']:
        raise ValueError('Missing/mismatched modal fingerprint in artifact')


def semantic_tests(full, brief, processed, output=None):
    """Real observation tests; preserve modes, cfg, and RNG exactly."""
    from libero.lifelong.runbook_core import rng_state, restore_rng
    import copy
    saved_rng = rng_state()
    modes = [(m, m.training) for root in (full, brief) for m in root.modules()]
    flag = full.cfg.policy.use_state
    try:
        full.eval()
        brief.eval()
        with torch.no_grad():
            h = full.spatial_encode(processed)
            b, t, _, e = h.shape
            instr = processed['task_emb']
            masks = processed['masks'].squeeze(1)
            language = full.language_encoder(instr, masks)
            langflat = language[:, None, :].expand(-1, t, -1).reshape(b * t, e)
            state = torch.cat([processed['obs'][k] for k in ['joint_states', 'gripper_states']], -1)
            statefeat = full.state_encoder(state.reshape(b * t, 9)).reshape(b, t, 1, e)
            expected = ((1 + full.state_scale(langflat)).reshape(b, t, 1, e) * statefeat
                        + full.state_shift(langflat).reshape(b, t, 1, e))
            torch.testing.assert_close(h[:, :, 3:4], expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(h[:, :, 0], language[:, None].expand(-1, t, -1))
            for index, key in enumerate(['agentview_rgb', 'eye_in_hand_rgb'], 1):
                img = processed['obs'][key].reshape(b*t, *processed['obs'][key].shape[2:])
                vision = full.image_proj(full.image_encoders(img, langs=None))
                vision = ((1 + full.film_scale(langflat)) * vision + full.film_shift(langflat)).reshape(b, t, e)
                torch.testing.assert_close(h[:, :, index], vision, rtol=1e-5, atol=1e-6)
            deltas = []
            for amount in (.01, .1, .5):
                changed = copy.deepcopy(processed)
                for key in ['joint_states', 'gripper_states']:
                    changed['obs'][key] += amount
                shifted = full.spatial_encode(changed)
                assert torch.equal(h[:, :, :3], shifted[:, :, :3])
                delta = (h[:, :, 3] - shifted[:, :, 3]).abs().max().item()
                if delta <= 1e-6:
                    raise ValueError('State nonresponse: delta=%g, perturbation=%g' % (delta, amount))
                deltas.append(delta)
            full.cfg.policy.use_state = False
            negative0 = full.spatial_encode(processed)
            negative1 = full.spatial_encode(changed)
            assert torch.equal(negative0, negative1)
            full.cfg.policy.use_state = True
            a = full.policy_head.forward_fn(full.temporal_encode(h)[:, -1])
            bdist = brief.policy_head.forward_fn(brief.temporal_encode(h)[:, -1])
            errors = []
            for left, right in zip(a, bdist):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                errors.append((left - right).abs().max().item())
            report = {'status': 'PASS', 'real_observations_and_weights': True,
                'state_token_formula_max_error': (h[:, :, 3:4] - expected).abs().max().item(),
                'token_identity_order': ['language', 'agentview', 'eye_in_hand', 'state'],
                'state_perturbation_max_deltas': deltas, 'first3_tokens_exactly_unchanged': True,
                'false_branch_negative_control_max_delta': (negative0-negative1).abs().max().item(),
                'gmm_mean_scale_logits_max_errors': errors, 'negative_control_is_software_only': True}
            if output:
                atomic_json(output, report)
            return report
    finally:
        full.cfg.policy.use_state = flag
        for module, training in modes:
            module.training = training
        restore_rng(saved_rng)
