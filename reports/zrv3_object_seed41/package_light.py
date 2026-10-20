"""Explicit allowlist for lightweight scientific metadata, never model/RNG/payload."""
import json, shutil, hashlib, tarfile
from server_runner import N, paths, js
L=N/'light';L.mkdir(exist_ok=True);manifest={}
def take(p,rel):
    if not p.is_file():return
    assert p.suffix in ('.json','.jsonl','.exit','.log','.md','.py','.yaml','.txt')
    assert not any(x in rel.split('/') for x in ('buffer','features','models','recovery'))
    assert p.stat().st_size<8*1024**2
    raw=p.read_bytes();raw.decode('utf-8')
    q=L/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
    manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for name in ('ORCHESTRATION.json','RUN_STATUS.json','RESULTS.json','BRANCH_DECISION.json','FINAL_ACCEPTANCE.json','PROCESS_EXIT_ACCEPTANCE.json','BLOCKED.json','controller.exit','supervisor.exit','controller.log','PROTOCOL.md'):
    take(N/name,'current/'+name)
for v in ('v3','v3a'):
    W,O=paths(v)
    if not O.exists():continue
    for p in O.glob('*.json'):take(p,v+'/'+p.name)
    for p in O.glob('*.exit'):take(p,v+'/'+p.name)
    for p in O.glob('*.log'):take(p,v+'/'+p.name)
    for name in ('STRUCTURE.jsonl','commands.jsonl','timings.jsonl'):
        take(O/name,v+'/'+name)
    for p in (O/'preflight').glob('*'):
        if p.suffix in ('.log','.exit','.jsonl'):take(p,v+'/preflight/'+p.name)
    for p in (O/'evaluation').glob('*'):
        if p.suffix in ('.json','.jsonl'):take(p,v+'/evaluation/'+p.name)
    take(W/'RUN_SPEC.json',v+'/RUN_SPEC.json')
    # Source records of existing controls; no new rollout.
    s=js(W/'RUN_SPEC.json')
    for label,root in [('RA',__import__('pathlib').Path(s['parent'])/'ra_coreset/evaluation'),('ZR_v2',__import__('pathlib').Path(s['old_zrv2_output'])/'evaluation')]:
        for p in root.glob('trials*.jsonl'):take(p,'controls/'+label+'/'+p.name)
(L/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(N/'light.tar.gz','w:gz') as t:
    for rel in list(manifest)+['PACKAGE_MANIFEST.json']:t.add(L/rel,arcname=rel)
print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),archive_sha256=hashlib.sha256((N/'light.tar.gz').read_bytes()).hexdigest())))
