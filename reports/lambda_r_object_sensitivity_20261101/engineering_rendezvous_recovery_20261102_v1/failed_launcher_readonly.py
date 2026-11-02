import json
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/rendezvous_recovery_20261102_v1')
print(json.dumps(dict(status='READ_ONLY_FAILED_LAUNCHER_PROBE',logs={n:(D/n).read_text() for n in ['A.log','B.log']},binding=json.loads((D/'BINDING.json').read_text()),validation_started=(D/'DISPATCH.json').exists(),ledger_exists=(D/'LEDGER.json').exists())))
