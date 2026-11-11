from pathlib import Path
import hashlib,json
root=Path('/data/mlr_ifa/experiment_workspaces/ABLATION_COMMON_DEBR_EVAL_20261111_v1')
spec=json.loads((root/'SPEC.json').read_text())
rows={}
for name,case in spec['evaluation_cases'].items():
 ref=Path(case['reference_dir'])
 file=ref/'evaluate.py'
 raw=file.read_bytes()
 contract=json.loads((ref/'CONTRACT.json').read_text())
 cpath=Path(case['config_root'])/'resolved_eval.json'
 rows[name]=dict(eval_sha=hashlib.sha256(raw).hexdigest(),config_equal=json.loads(cpath.read_text())==case['eval_config'],config_bound=contract['files'].get(str(cpath)),actual_eval_sha=contract['files'].get(str(file)),has_crlf=b'\r\n' in raw)
 print_text=raw.decode()
print(json.dumps(dict(references=rows,source=print_text,formal=0)))
