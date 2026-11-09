"""Reproduce the original Random selection, bind it, then link read-only payloads."""
import json,random,time,math
from pathlib import Path
from .common import O,SEED,sha,write,hashes
from libero.lifelong.algos import mlr
def store(c,t,benchmark):
 assert c.selector=='random'
 root=Path(c.run_root);assert not (root/'storage'/('task%d.json'%t)).exists()
 spec=json.loads((O/'INPUT.json').read_text());old=spec['storage'][str(t)]
 paths=mlr._list_pt(str(O/'features'/str(t)));k=max(1,math.ceil(.5*len(paths)))
 selected=mlr._sample(paths,k,random.Random(SEED))
 assert k==old['K'] and [Path(p).name for p in selected]==[Path(r['source']).name for r in old['selected']]
 before=hashes(Path(c.buffer_dir));assert before==old['before']
 selected_rows=[]
 for p,r in zip(selected,old['selected']):
  assert sha(p)==r['sha256'];dst=Path(c.buffer_dir)/r['destination'];assert not dst.exists()
  selected_rows.append(dict(source=p,destination=dst.name,sha256=r['sha256']))
 manifest=dict(task=t,K=k,current_count=len(paths),before=before,selected=selected_rows,selection=dict(selector='random',same_original_order=True),contract_sha256=sha(O/'CONTRACT.json'),modal_sha256=sha(O/'MODAL_CONTRACT.json'))
 write(root/'storage'/('task%d.json'%t),manifest)
 for p,r in zip(selected,old['selected']):(Path(c.buffer_dir)/r['destination']).symlink_to(Path(p).resolve())
 after=hashes(Path(c.buffer_dir));expected=dict(before);expected.update({r['destination']:r['sha256'] for r in old['selected']});assert after==expected
 write(root/('buffer_after_task%d.json'%t),dict(counts={str(j):sum(n.startswith('t%d__'%j) for n in after) for j in range(6,t+1)},total=len(after),hashes=after,linked_parent_payloads_readonly=True,modal_sha256=sha(O/'MODAL_CONTRACT.json')))
