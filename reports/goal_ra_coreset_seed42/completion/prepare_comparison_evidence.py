"""Copy already available, hash-verified lightweight baseline evidence; never remote execution."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;old=R.parent/'20261018_sc_coreset_goal_seed42'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contract=js(R/'final_evidence/CONTRACT.json');parent=js(R/'final_evidence/PARENT_BINDING.json')['parent']
for p in (old/'baseline_evidence').rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(old/'baseline_evidence');assert p.suffix in {'.json','.jsonl'}
 assert sha(p)==contract['files'][parent+'/'+rel.as_posix()]
 dst=R/'baseline_evidence'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
print('PASS: hash-verified original Random and Coreset evidence')
