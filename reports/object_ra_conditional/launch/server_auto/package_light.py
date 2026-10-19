"""Allowlisted metadata only. Never weight, payload, raw action or language-array copies."""
import json,time,hashlib
from pathlib import Path
from flow_support import N,js,write
def main():
 dest=N/('light_'+time.strftime('%Y%m%d_%H%M%S'));dest.mkdir();files={};excluded={}
 def collect(root,prefix):
  for p in root.rglob('*'):
   if not p.is_file() or p.suffix not in ['.json','.jsonl','.log','.exit','.md']:continue
   rel=p.relative_to(root)
   if any(x in rel.parts for x in ['source','attachment','buffer','features','models','recovery','bindings','original_failure','zr_template','zr_runs','zr_ledger']) or any(x.startswith('light_') for x in rel.parts):continue
   if p.name in ['calibration_batches.jsonl']:excluded[prefix+'/'+str(rel)]='Large full trace retained server-side; verified by audit';continue
   raw=p.read_bytes()
   if len(raw)>8*1024**2:excluded[prefix+'/'+str(rel)]='over8MiB; retained server-side';continue
   if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
   if b'"last_action"' in raw or b'"password"' in raw:excluded[prefix+'/'+str(rel)]='disallowed raw payload or credential field';continue
   name=prefix+'/'+rel.as_posix();q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),source=str(p))
 collect(N,'controller')
 for seed in [41,42,43]:
  o=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed)
  if o.exists():collect(o,'OBJECT_RA_seed%d'%seed)
 for s in (N/'zr_runs').glob('*/RUN_SPEC.json'):
  spec=js(s);collect(Path(spec['output']),s.parent.name)
 write(dest/'PACKAGE_MANIFEST.json',dict(time=time.time(),status=js(N/'STATUS.json')['status'],files=files,excluded=excluded))
 print(dest)
if __name__=='__main__':main()
