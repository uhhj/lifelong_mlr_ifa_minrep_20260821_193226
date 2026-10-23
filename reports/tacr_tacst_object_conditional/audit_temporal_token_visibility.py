"""Independently bind frozen source, learned-checkpoint probe and readout indexing."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('probe')
ap.add_argument('source_capture')
a = ap.parse_args()
p, s = Path(a.probe), Path(a.source_capture)
e = json.loads((p/'EVIDENCE.json').read_text())
c = json.loads((s/'EVIDENCE.json').read_text())
assert e['status'] == 'FROZEN_TEMPORAL_MODALITY_VISIBILITY_PROVEN_ZERO_TRAINING'
assert c['status'] == 'ACTUAL_FROZEN_TOKEN_SOURCE_CAPTURED'
assert e['frozen_files'] == c['frozen_files'] == 1462
assert e['frozen_before_after_unchanged'] and c['frozen_hashes_unchanged']
for key, file in [('brief', 'libero/lifelong/models/brief.py'),
                  ('complete', 'libero/lifelong/models/complete_policy.py'),
                  ('transformer', 'libero/lifelong/models/modules/transformer_modules.py'),
                  ('physical_branch', 'caet_v3_exp/model.py')]:
    assert e['sources'][key]['sha256'] == c['files'][file]['sha256']

def segment(file, name):
    return next(x['source'] for x in c['files'][file]['segments'] if x['name'] == name)

def compact(text):
    return ''.join(text.split())

for file in ('libero/lifelong/models/brief.py', 'libero/lifelong/models/complete_policy.py'):
    text = compact(segment(file, 'temporal_encode'))
    assert 'TensorUtils.join_dimensions(x,1,2)' in text
    assert 'out.view(B,T,M,E)' in text and 'returnout[:,:,0,:]' in text
assert 'returnp.temporal_encode(feature(x))[:,-1]' in compact(segment('caet_v3_exp/model.py', 'hidden'))
assert 'torch.cat([lang_tok,vision_mod,state_mod],dim=2)' in compact(segment('libero/lifelong/models/complete_policy.py', 'spatial_encode'))
assert 'attention_mask=attention_mask' in compact(segment('libero/lifelong/models/modules/transformer_modules.py', 'OfficialGPT2Decoder'))
assert all(e['config_policy_flags'].values())
assert e['all_six_layers_standard_causal_mask'] and 'torch.where(causal_mask' in e['actual_attention_source']
selected = (e['input_shape'][1]-1)*e['input_shape'][2]
assert selected == e['readout_flat_index'] == 28 and e['full_last_index'] == 31
assert e['selected_visible_input_indices'] == list(range(selected+1))
assert e['selected_invisible_input_indices'] == [29, 30, 31]
by = {r['input_flat_index']: r for r in e['perturbations']}
for i in (29, 30, 31):
    assert by[i]['selected28_max_abs_change'] == by[i]['learned_R_max_abs_change'] == by[i]['learned_effect_fixed_action_max_abs_change'] == 0
    assert by[i]['full_last31_max_abs_change'] > 0
for i in (25, 26, 27):
    assert by[i]['selected28_max_abs_change'] > 0
assert e['input_gradient_l2_by_time_modality'][7][1:] == [0, 0, 0]
assert all(x > 0 for x in e['input_gradient_l2_by_time_modality'][6][1:])
assert e['selected_ignores_joint_current_visual_state_perturbation']
assert e['one_step_selected_ignores_visual_state_max_abs_change'] == 0
assert e['input_gradient_calls'] == 1 and not e['parameter_gradients_created']
assert all(e[k] == c[k] == 0 for k in ('optimizer_updates', 'environment_steps', 'server_writes'))
assert not e['cuda_initialized'] and not e['physical_generalization_or_effect_quality_measured']
result = dict(status='LOCAL_FROZEN_TOKEN_READOUT_VISIBILITY_AUDIT_PASS',
              observed_at=datetime.datetime.now().astimezone().isoformat(),
              probe_sha256=hashlib.sha256((p/'EVIDENCE.json').read_bytes()).hexdigest(),
              source_capture_sha256=hashlib.sha256((s/'EVIDENCE.json').read_bytes()).hexdigest(),
              selected_flat_index=28, invisible_latest_modalities=[1,2,3],
              current_physical_modalities_structurally_unreachable=True,
              historical_physical_modalities_accessible=True,
              temporal_forward_calls=e['model_forwards'],
              temporal_counter_excludes_R_E_D_component_calls=True,
              input_only_vector_jacobian_products=1,
              full_input_jacobian_not_explicitly_materialized=True,
              no_physical_prediction_quality_or_causal_return_effect_measured=True,
              no_optimizer_environment_cuda_or_server_writes=True,
              no_frozen_experiment_change=True,
              evidence_scope='Learned TACR41 stage9 checkpoint on synthetic latent perturbations plus actual frozen source and masks. Shared RA/ALIGN/RST indexing is a structural code finding, not measured downstream improvement.')
(p/'LOCAL_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
