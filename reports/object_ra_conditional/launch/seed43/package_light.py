"""Metadata-only snapshot. Never transfers real action targets, tensors or credentials."""
from pathlib import Path
import os,json,time,hashlib
O=Path(os.environ['CORESET_OUT']);W=Path(__file__).resolve().parent
def main():
    out=W/('light_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();manifest={}
    for p in sorted(O.rglob('*')):
        if not p.is_file() or p.suffix not in {'.json','.jsonl','.log','.exit','.pid','.started'}:continue
        if any(x in p.parts for x in ['features','buffer','models','recovery','smoke','bindings']):continue
        rel=p.relative_to(O);dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes()
        if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
        dst.write_bytes(raw);manifest[str(rel)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    for p in (O/'bindings').glob('*.json'):
        obj=json.loads(p.read_text())
        for r in obj['windows'].values():del r['last_action']
        obj['raw_server_manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();obj['last_actions_omitted_for_publication']=True
        dst=out/'binding_metadata'/p.name;dst.parent.mkdir(exist_ok=True);raw=json.dumps(obj,indent=2).encode();dst.write_bytes(raw)
        manifest[str(dst.relative_to(out))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (out/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(status='PARTIAL_UNLESS_PROCESS_EXIT_ACCEPTANCE_PASS',time=time.time(),files=manifest),indent=2));print(out)
if __name__=='__main__':main()
