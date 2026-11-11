"""Use the prepared reverse namespace verifier for the three GOAL MLR cases."""
import ast
import json
import os
from pathlib import Path
import runpy
import sys

from reverse_common import Q, read, sha, verify_case


def verifier_node(path):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify')


def install(case):
    assert case in ('GOAL_MLR_41', 'GOAL_MLR_42', 'GOAL_MLR_43')
    directory = Q / 'cases' / case
    inputs = read(directory / 'INPUT.json')
    bound_contract = verify_case(directory)
    template = Q / 'cases' / 'OBJECT_MLR_41'
    verify_case(template)
    old_path = directory / 'source/coreset_exp/common.py'
    new_path = template / 'source/coreset_exp/common.py'
    legacy = verifier_node(old_path)
    replacement = verifier_node(new_path)
    assert legacy.body[2].targets[0].id == 'orchestration'
    assert 'orchestration_sha256' in ast.dump(legacy.body[3])
    assert isinstance(legacy.body[4], ast.For) and 'orchestration' in ast.dump(legacy.body[4])
    legacy.body[2:5] = []
    assert ast.dump(legacy) == ast.dump(replacement), 'Namespace verification differs beyond the legacy orchestration manifest'

    os.environ.update(REVERSE_CASE_ROOT=str(directory), CORESET_OUT=inputs['input_root'], CORESET_SEED=str(inputs['seed']))
    sys.path[:0] = [str(directory / 'source'), str(directory), str(Q)]
    from order import install as install_order
    install_order()
    from coreset_exp import common
    assert Path(common.__file__).resolve() == old_path
    namespace = common.__dict__.copy()
    function = ast.Module(body=[replacement], type_ignores=[])
    exec(compile(function, str(new_path), 'exec'), namespace)
    common.verify = namespace['verify']
    return common, dict(case=case, case_contract=bound_contract, frozen_GOAL_source_sha256=sha(old_path),
                        reference_OBJECT_source_sha256=sha(new_path), legacy_orchestration_statements_removed=3)


if __name__ == '__main__':
    case, phase = sys.argv[1:]
    common, proof = install(case)
    if phase == 'verify-only':
        import torch
        common.verify()
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_ACTUAL_REVERSE_NAMESPACE_VERIFICATION', **proof,
                              new_optimizer=0, new_environments=0)))
    else:
        assert phase == 'train'
        sys.argv = [str(Q / 'run_case.py'), case, phase]
        runpy.run_path(sys.argv[0], run_name='__main__')
