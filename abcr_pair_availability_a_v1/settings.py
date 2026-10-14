"""One shared configuration for the supplied probe and batch supplement."""
from types import SimpleNamespace

FROZEN_PARAMETERS = dict(
    past_q=1, visual_thresholds=[0.7, 0.8, 0.9, 0.95], action_share=0.5,
    action_bif=1.5, topk=5, batch_new=5, batch_old=5, batch_trials=500,
    max_current=0, max_old_per_task=0, audit_per_bucket=15, chunk=128,
)


def validate(parameters):
    assert set(parameters) == set(FROZEN_PARAMETERS), 'PARAMETER_FIELDS_MISMATCH'
    assert parameters['action_share'] < parameters['action_bif']
    for key in ('past_q', 'topk', 'batch_new', 'batch_old', 'batch_trials', 'chunk'):
        assert isinstance(parameters[key], int) and parameters[key] > 0
    assert parameters['visual_thresholds'] == sorted(set(parameters['visual_thresholds']))
    assert all(-1 <= v <= 1 for v in parameters['visual_thresholds'])


def native_args(stage, output, parameters):
    validate(parameters)
    args = dict(parameters)
    args['visual_thresholds'] = ','.join(str(x) for x in parameters['visual_thresholds'])
    args.update(feature_dir=stage['feature_dir'], buffer_dir=stage['buffer_dir'],
        current_task=stage['task'], seed=stage['seed'], out_dir=str(output))
    result = SimpleNamespace(**args)
    assert_consistent(result, parameters)
    return result


def assert_consistent(args, parameters):
    for key, value in parameters.items():
        actual = getattr(args, key)
        if key == 'visual_thresholds':
            actual = [float(x) for x in actual.split(',')]
        assert actual == value, 'PROBE_BATCH_PARAMETER_MISMATCH:' + key
