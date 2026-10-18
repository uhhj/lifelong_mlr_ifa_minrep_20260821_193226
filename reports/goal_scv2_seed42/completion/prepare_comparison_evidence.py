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
src=old/'final_evidence';manifest=js(src/'PACKAGE_MANIFEST.json')['files'];proof={}
selected=[src/'AUDIT.json',src/'PROCESS_EXIT_ACCEPTANCE.json']+list((src/'sc_coreset/evaluation').glob('*.jsonl'))
for p in selected:
 rel=p.relative_to(src);assert sha(p)==manifest[rel.as_posix()]['sha256']
 dst=R/'scv1_evidence'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
 proof[rel.as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size,source=str(p))
(R/'scv1_evidence/PROVENANCE.json').write_text(json.dumps(dict(source_package=str(src/'PACKAGE_MANIFEST.json'),source_package_sha256=sha(src/'PACKAGE_MANIFEST.json'),files=proof),indent=2)+'\n',encoding='utf-8')
print('PASS: copied verified read-only baseline metadata and SC-v1 outcome evidence')
