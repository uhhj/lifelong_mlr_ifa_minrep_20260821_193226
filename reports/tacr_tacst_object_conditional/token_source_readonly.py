"""Capture actual frozen readout and token-construction source without imports."""
import ast
import hashlib
import json
import time
from pathlib import Path
Q = Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
S = Q/'TACR_41/source'
targets = {
    'libero/lifelong/models/brief.py': ['temporal_encode', 'forward'],
    'libero/lifelong/models/complete_policy.py': ['spatial_encode', 'temporal_encode', 'get_action'],
    'libero/lifelong/models/modules/transformer_modules.py': ['OfficialGPT2Decoder'],
    'caet_v3_exp/model.py': ['feature', 'hidden', 'DeployResidual'],
    'coreset_exp/model.py': ['cfg', 'complete', 'strict_downstream'],
}
files = {}
for name, wanted in targets.items():
    p = S/name
    raw = p.read_bytes()
    source = raw.decode()
    segments = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            segments.append(dict(name=node.name, line=node.lineno, source=ast.get_source_segment(source, node)))
    assert set(wanted) <= {r['name'] for r in segments}
    files[name] = dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), segments=segments)
contract = json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in contract['files'].items())
print(json.dumps(dict(status='ACTUAL_FROZEN_TOKEN_SOURCE_CAPTURED', observed_unix=time.time(),
                      files=files, frozen_files=len(contract['files']), frozen_hashes_unchanged=True,
                      model_forwards=0, optimizer_updates=0, environment_steps=0, server_writes=0)))
