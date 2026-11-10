"""Materialize only reviewed text metadata returned by CPU preparation."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys
R=Path(__file__).resolve().parent
snap=Path(sys.argv[1]);e=json.loads((snap/'EVIDENCE.json').read_text())
assert e['status']=='PASS_PREPARED_FROZEN_NOT_LAUNCHED'
sha=lambda b:hashlib.sha256(b).hexdigest()
files={}
for name,row in e['diagnostic_files'].items():
    p=snap/name;raw=p.read_bytes();assert sha(raw)==row['sha256']
    item=json.loads(raw);rel=PurePosixPath(item['relative'])
    assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix in ['.json','.yaml','.yml']
    raw=item['text'].encode('utf-8');assert sha(raw)==item['sha256'] and len(raw)<16*1024**2
    if rel.suffix=='.json':json.loads(raw)
    dest=R/str(rel);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    files[str(rel)]=dict(sha256=sha(raw),bytes=len(raw))
assert files['CONTRACT.json']['sha256']==e['contract_sha256']
receipt=dict(status='PASS_EXACT_PREPARED_PUBLIC_TEXT_EXTRACTED',evidence=str(snap),contract_sha256=e['contract_sha256'],files=files,private_payloads_transferred=False)
(R/'PREPARATION_PUBLIC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],files=len(files),bytes=sum(x['bytes'] for x in files.values()),contract=e['contract_sha256'])))
