"""External audit strengthening only; never edits any deployed/frozen source."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent;scripts={};changes={}
for p in sorted(R.glob('final_*_readonly.py')):
 s=p.read_text(encoding='utf8');before=hashlib.sha256(p.read_bytes()).hexdigest()
 marker="manifests={};stage_labels={}"
 extra="""parent_binding=js(Q/'PARENT_BINDING.json')
assert all(sha(p)==h for p,h in parent_binding['files'].items())
initial_record=js(Q/'INITIAL_C.json');official_record=js(O/'ra_coreset/INITIAL_STATE.json')
assert initial_record['tensors']==official_record['tensors']
assert all(initial_record['rng'][k]==official_record['rng'][k] for k in initial_record['rng'] if k!='cuda')
manifests={};stage_labels={}"""
 if 'parent_binding=js(' not in s:s=s.replace(marker,extra)
 ast.parse(s,feature_version=(3,8));p.write_text(s,encoding='utf8');scripts[p.stem[len('final_'):-len('_readonly')]]=s
 changes[p.name]=dict(before=before,after=hashlib.sha256(p.read_bytes()).hexdigest())
assert len(scripts)==5
(R/'final_readonly.py').write_text('SCRIPTS='+ascii(scripts)+'\n'+(R/'final_template.py').read_text(encoding='utf8'),encoding='utf8')
(R/'FINAL_WRAPPER_STATIC_REVIEW.json').write_text(json.dumps(dict(status='AST_ONLY_PENDING_ACTUAL_FINAL_ACCEPTANCE',changes=changes,remote_writes=0,model_calls=0,optimizer_updates=0,environments=0),indent=2),encoding='utf8')
