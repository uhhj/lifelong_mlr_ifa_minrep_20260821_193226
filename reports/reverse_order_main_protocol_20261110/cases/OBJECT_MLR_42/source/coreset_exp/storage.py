import json,random,math,shutil,time
from pathlib import Path
import numpy as np
import h5py
from .common import O,sha,write,hashes
from libero.lifelong.algos import mlr
from libero.lifelong.selectors.greedy_kcenter import greedy_kcenter_select

def store(c,t,benchmark):
    start=time.monotonic();root=Path(c.run_root);mfile=root/'storage'/f'task{t}.json'
    paths=mlr._list_pt(str(O/'features'/str(t)));k=max(1,math.ceil(.5*len(paths)))
    if mfile.exists():
        manifest=json.loads(mfile.read_text());assert manifest['contract_sha256']==sha(O/'CONTRACT.json')
        assert manifest['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
    else:
        before=hashes(Path(c.buffer_dir));assert not any(n.startswith(f't{t}__') for n in before)
        assert len(before)+k<=mlr.BUFFER_CAPACITY,'Unexpected capacity branch; stop before changing quotas'
        if c.selector=='random':selected=mlr._sample(paths,k,random.Random(42));report={'selector':'random'}
        else:
            demo_ids=json.loads((O/'FEATURES.json').read_text())['demos'][str(t)]
            with h5py.File(str(Path(c.folder)/benchmark.get_task_demonstration(t)),'r') as f:
                population=np.concatenate([f['data'][d]['actions'][:] for d in demo_ids])
            selected,report=greedy_kcenter_select(paths,k,42,population)
        assert len(selected)==len(set(selected))==k
        manifest=dict(task=t,K=k,current_count=len(paths),before=before,
            selected=[dict(source=p,destination=f't{t}__{Path(p).name}',sha256=sha(p)) for p in selected],
            selection=report,contract_sha256=sha(O/'CONTRACT.json'),modal_sha256=sha(O/'MODAL_CONTRACT.json'))
        write(mfile,manifest) # Precommit before copies: task-boundary retries cannot resample.
    buffer=Path(c.buffer_dir);buffer.mkdir(parents=True,exist_ok=True)
    allowed=dict(manifest['before']);allowed.update({r['destination']:r['sha256'] for r in manifest['selected']})
    actual=hashes(buffer);assert set(actual)<=set(allowed) and all(allowed[n]==h for n,h in actual.items())
    for r in manifest['selected']:
        src=Path(r['source']);dst=buffer/r['destination'];assert sha(src)==r['sha256']
        if dst.exists():assert sha(dst)==r['sha256']
        else:
            tmp=dst.with_suffix('.copying');shutil.copy2(src,tmp);assert sha(tmp)==r['sha256'];tmp.replace(dst)
    after=hashes(buffer);assert after==allowed
    counts={str(i):sum(n.startswith(f't{i}__') for n in after) for i in range(6,t+1)}
    result=dict(counts=counts,total=len(after),hashes=after,modal_sha256=sha(O/'MODAL_CONTRACT.json'),
                seconds=time.monotonic()-start,metadata_bytes=mfile.stat().st_size)
    if c.selector=='coreset':assert counts==json.loads((O/'random'/f'buffer_after_task{t}.json').read_text())['counts']
    write(root/f'buffer_after_task{t}.json',result)
    return result
