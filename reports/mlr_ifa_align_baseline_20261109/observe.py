"""Read only small logs and metadata; never serialize model/state payloads."""
from runtime import *
def snapshot():
 c=verify();light={}
 def add(p,rel):
  raw=p.read_bytes()
  if p.suffix=='.jsonl':raw=raw[:raw.rfind(b'\n')+1]
  light[rel]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 for name in ['controller.exit','STATUS.json','EVENTS.jsonl','FAILURE.json','DEPLOYMENT.json','OBJECT_COMPLETE.json','GOAL_COMPLETE.json']:
  if (Q/name).exists():add(Q/name,'coordinator/'+name)
 for case in CASES:
  d=P/case
  for p in d.rglob('*'):
   if not p.is_file() or p.is_symlink() or p.suffix not in ['.json','.jsonl','.exit']:continue
   rel=p.relative_to(d)
   if rel.parts[0] in ['source','features','manifests'] or p.name in ['INPUT.json','CONTRACT.json','MODAL_CONTRACT.json','FEATURES.json','resolved_random.json','resolved_eval.json']:continue
   add(p,case+'/'+rel.as_posix())
 return dict(status='READONLY_PROGRESS_NOT_FINAL',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),processes=processes([Q,P]),light=light,new_optimizer=0,new_environments=0)
if __name__=='__main__':print(json.dumps(snapshot()))
