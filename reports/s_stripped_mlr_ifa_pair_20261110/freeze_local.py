"""Freeze reviewed UTF-8/LF text; this script never touches the server."""
from pathlib import Path
import json,hashlib,ast,datetime
R=Path(__file__).resolve().parent
def text(p):return p.read_text(encoding='utf-8-sig').replace('\r\n','\n')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
root_files=['runtime.py','controller.py','check_readonly.py','final_case_cpu.py','final_readonly.py']
files={n:text(R/n) for n in root_files}
for p in sorted((R/'cases').rglob('*')):
 if p.is_file() and p.suffix in ['.py','.json']:files[p.relative_to(R).as_posix()]=text(p)
for n,s in files.items():
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
 elif n.endswith('.json'):json.loads(s)
assert not (R/'ACTIVE_ROOT.json').exists(),'Already deployed; cannot refreeze'
external=json.loads((R/'EXTERNAL_FILES.json').read_text(encoding='utf-8'))
contract=dict(name='S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1',created=datetime.datetime.now().astimezone().isoformat(),code_files={n:digest(s) for n,s in files.items()},external_files=external,objective='MLR: original_BC+0.1*IFA; ALIGN: aligned_BC+0.1*IFA',sampling='Frozen original Random membership/order and Random IFA task pairs',evaluation='Full-S cached frame, per-trajectory fixed Normal, zero-done lanes, fresh vector environment',cases=json.loads((R/'PREPARATION_STATUS.json').read_text())['cases'],production=1089600,accepted_updates=1089600,admission_optimizer_calls=240,training_collection=0,formal=2400,epochs=4800,stages=[6,7,8,9],epochs_each=100,max_parallel_training=2,queue='OBJECT six training -> OBJECT 1200 formal -> GOAL six training -> GOAL 1200 formal')
raw=json.dumps(contract,ensure_ascii=False,indent=2)+'\n';files['CONTRACT.json']=raw
(R/'CONTRACT.json').write_text(raw,encoding='utf-8',newline='\n')
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(files,ensure_ascii=False),encoding='utf-8',newline='\n')
review=dict(status='FROZEN_TEXT_NOT_DEPLOYED',contract_sha256=digest(raw),files_sha256={n:dict(sha256=digest(s),bytes=len(s.encode())) for n,s in files.items()},external_files=len(external),payload='Only Python and JSON metadata; no tensor/model/demonstration/RNG payload')
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=review['status'],contract=review['contract_sha256'],files=len(files))))
