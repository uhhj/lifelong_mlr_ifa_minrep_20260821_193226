"""Split oversized metadata snapshots without removing any original evidence."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
def prepare():
 excluded=set()
 for p in (R/'engineering_cached_eval_20261110_v1').glob('*/EVIDENCE.json'):
  raw=p.read_bytes()
  if len(raw)<8*1024**2:continue
  e=json.loads(raw);assert 'light' in e
  dest=p.parent/'PUBLICATION_SHARDS';dest.mkdir(exist_ok=True)
  groups=[];current={};size=0
  for k,v in e['light'].items():
   n=len(json.dumps({k:v}).encode());assert n<4*1024**2
   if current and size+n>4*1024**2:groups.append(current);current={};size=0
   current[k]=v;size+=n
  if current:groups.append(current)
  shards={};reconstructed={}
  for i,g in enumerate(groups):
   name=f'light_{i:02d}.json';payload=json.dumps(g,ensure_ascii=True,separators=(',',':')).encode()+b'\n'
   (dest/name).write_bytes(payload);shards[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload));reconstructed.update(json.loads(payload))
  assert reconstructed==e['light']
  header={k:v for k,v in e.items() if k!='light'}
  index=dict(status='LOSSLESS_LIGHT_METADATA_SPLIT',original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),header=header,light_shards=shards,light_items=len(reconstructed),payloads='metadata, hashes, scalar logs and outcomes only; no tensor or RNG state')
  (dest/'INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf8')
  excluded.add(p)
 return excluded
if __name__=='__main__':print(len(prepare()))
