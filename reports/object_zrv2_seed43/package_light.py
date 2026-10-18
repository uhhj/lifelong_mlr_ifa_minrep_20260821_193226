"""Snapshot explicitly permitted metadata only. Never includes tensors or archives."""
import json,hashlib,time,shutil
from pathlib import Path
O=Path('/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018')
W=Path('/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    stamp=time.strftime('%Y%m%d_%H%M%S');dst=W/('light_'+stamp);dst.mkdir()
    allowed={'.json','.jsonl','.log','.exit','.pid','.started'};manifest={}
    for p in sorted(O.rglob('*')):
        if not p.is_file() or p.suffix not in allowed or any(x in p.parts for x in ['buffer','models','recovery']):continue
        rel=p.relative_to(O);out=dst/rel;out.parent.mkdir(parents=True,exist_ok=True)
        # Reading an append-only log yields an evidence prefix. Drop only a partial last line in the copy.
        raw=p.read_bytes()
        if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
        out.write_bytes(raw);manifest[str(rel)]=dict(bytes=len(raw),sha256=sha(out),source=str(p))
    (dst/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(time=time.time(),status='PARTIAL_UNLESS_FINAL_EXIT_ACCEPTANCE_PRESENT',files=manifest),indent=2))
    print(dst)
if __name__=='__main__':main()
