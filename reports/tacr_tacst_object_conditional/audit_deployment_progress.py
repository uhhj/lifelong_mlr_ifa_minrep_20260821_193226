"""Bind current formal metadata to CPU-verified deployment hashes; no new evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from formal_audit import audit_formal

ap = argparse.ArgumentParser()
ap.add_argument('snapshot')
ap.add_argument('checkpoints')
a = ap.parse_args()
p, c = Path(a.snapshot), Path(a.checkpoints)
e = json.loads((p/'EVIDENCE.json').read_text())
d = json.loads((c/'EVIDENCE.json').read_text())
assert d['status'] == 'CPU_SCHEMA_AND_STRICT_RESIDUAL_STUDENT_LOAD_PASS_NOT_FINAL_ACCEPTANCE'
assert d['frozen_files'] == 1462 and d['frozen_before_after_unchanged']
assert all(d[k] == 0 for k in ('model_forwards', 'optimizer_updates', 'environment_steps', 'server_writes'))
assert not d['cuda_initialized'] and not e['frozen_hashes']['failures']
run = e['runs'][d['key']]
f = run['formal']
formal = audit_formal(f, run['entry']['seed'])
by = {r['task']: r for r in d['checkpoints']}
assert set(by) == {6, 7, 8, 9}
all_rows = list(f['trials']) + list(f['attempts']) + list(f['cells'].values())
stages = set()
for row in all_rows:
    proof = by[row['stage']]
    assert row['epoch'] == 99
    assert row['loaded_downstream_keys'] == 100 and row['loaded_residual_keys'] == 14
    assert row['checkpoint_sha256'] == proof['main_checkpoint_sha256']
    assert row['deployment_residual_sha256'] == proof['deployment_sha256']
    stages.add(row['stage'])
result = dict(status='LOCAL_DEPLOYMENT_PROGRESS_BINDING_PASS_NOT_FINAL_ACCEPTANCE',
              key=d['key'], formal=formal, formal_stages_with_strict_main_load_records=sorted(stages),
              four_main_checkpoints_schema_keys=100, four_residuals_strict_keys=14,
              four_students_strict_keys=20, current_rows_bound_to_actual_hashes=len(all_rows),
              main_strict_loading_for_unvisited_stages_pending=True,
              snapshot_sha256=hashlib.sha256((p/'EVIDENCE.json').read_bytes()).hexdigest(),
              checkpoint_evidence_sha256=hashlib.sha256((c/'EVIDENCE.json').read_bytes()).hexdigest(),
              no_training_rollout_or_parameter_change=True)
(p/'DEPLOYMENT_PROGRESS_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
