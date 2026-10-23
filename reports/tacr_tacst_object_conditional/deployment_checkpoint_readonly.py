"""CPU-only checkpoint schema/deployment check during evaluation; no model forward."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import torch

torch.set_num_threads(1)
Q = Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
state = json.loads((Q/'RUN_STATUS.json').read_text())
key = state['current']
entry = json.loads((Q/key/'ENTRY.json').read_text())
O = Path(entry['output'])
root = O/'ra_coreset'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

contract = json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
frozen_before = [p for p, h in contract['files'].items() if sha(p) != h]
assert not frozen_before
spec = importlib.util.spec_from_file_location('deployment_readonly_model', Q/key/'source/caet_v3_exp/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parent = torch.load(Path(entry['parent'])/'ra_coreset/models/mlrifa_6_model_ep99.pth', map_location='cpu')['state_dict']
assert len(parent) == 100
records = []
for task in range(6, 10):
    cp = root/'models'/('mlrifa_%d_model_ep99.pth' % task)
    dp = root/'deployed_residual'/('task%d.pth' % task)
    sp = root/'aux_heads'/('task%d.pth' % task)
    meta = json.loads(dp.with_suffix('.json').read_text())
    raw = torch.load(cp, map_location='cpu')
    core = raw['state_dict']
    assert raw['task'] == task and raw['epoch_index'] == 99
    assert raw['contract_sha256'] == sha(O/'CONTRACT.json')
    assert len(core) == 100 and set(core) == set(parent)
    assert all(core[k].shape == parent[k].shape and core[k].dtype == parent[k].dtype for k in core)
    assert all(v.device.type == 'cpu' and torch.isfinite(v).all() for v in core.values())
    deploy_state = torch.load(dp, map_location='cpu')
    student_state = torch.load(sp, map_location='cpu')['model']
    assert len(deploy_state) == 14 and len(student_state) == 20
    deploy = module.DeployResidual().cpu()
    student = module.Student().cpu()
    deploy.load_state_dict(deploy_state, strict=True)
    student.load_state_dict(student_state, strict=True)
    assert all(torch.equal(v, student.deploy.state_dict()[k]) for k, v in deploy_state.items())
    assert meta['task'] == task and meta['deploy_keys'] == 14
    assert meta['sha256'] == sha(dp) and meta['main_checkpoint_sha256'] == sha(cp)
    assert meta['training_student_sha256'] == sha(sp)
    records.append(dict(task=task, main_checkpoint_sha256=sha(cp), main_keys=100,
                        main_schema_exact_parent_keys_shapes_dtypes=True,
                        main_model_instantiated_or_strict_loaded=False,
                        deployment_sha256=sha(dp), strict_loaded_deployment_keys=14,
                        student_sha256=sha(sp), strict_loaded_student_keys=20,
                        training_and_deployment_parameters_exact=True,
                        deployment_metadata=meta))
    del raw, core, deploy_state, student_state, deploy, student
assert not torch.cuda.is_initialized()
assert not [p for p, h in contract['files'].items() if sha(p) != h]
print(json.dumps(dict(status='CPU_SCHEMA_AND_STRICT_RESIDUAL_STUDENT_LOAD_PASS_NOT_FINAL_ACCEPTANCE',
                      key=key, observed_unix=time.time(), frozen_files=len(contract['files']),
                      frozen_before_after_unchanged=True, checkpoints=records,
                      main_strict_model_loading_requires_formal_or_final_audit=True,
                      model_forwards=0, optimizer_updates=0, environment_steps=0,
                      cuda_initialized=False, server_writes=0)))
