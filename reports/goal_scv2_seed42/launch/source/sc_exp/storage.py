import json,math,time,random,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch,h5py
from coreset_exp.common import O,SEED,SELECTOR_SEED,write,sha,hashes,rng,restore_rng
from libero.lifelong.algos import mlr
from libero.lifelong.selectors.greedy_kcenter import greedy_kcenter_select,Distance
from .selection import repair

def select(paths,k,population,binding):
    begin=time.monotonic();saved=rng()
    try:
        s0,original=greedy_kcenter_select(paths,k,SELECTOR_SEED,population)
        ordered=sorted(paths,key=lambda p:Path(p).name);ids=[Path(p).name for p in ordered]
        data=[torch.load(p,map_location='cpu') for p in ordered]
        vv=np.array([x['feature'][0,:,1:3,:].numpy() for x in data]);aa=np.array([x['actions'][0,-1,:].numpy() for x in data])
        for p,a in zip(ordered,aa):
            row=binding[Path(p).name];assert sha(p)==row['sha256']
            assert np.array_equal(a,np.asarray(row['last_action'],dtype=np.float32))
        d=Distance(vv,aa,population,SELECTOR_SEED);D=d.distances(np.arange(len(ids)))
        assert d.hs==original['visual_scale'] and d.as_==original['action_distance_scale']
        report=repair(ids,binding,[Path(p).name for p in s0],D)
        names=report['selected_ids']
        idx={n:p for n,p in zip(ids,ordered)}
        report.update(original_selection=original,original_ids=[Path(p).name for p in s0],selected_ids=names,
                      demo_full=dict(Counter(binding[n]['demo_id'] for n in ids)),
                      demo_original=dict(Counter(binding[Path(p).name]['demo_id'] for p in s0)),
                      demo_final=dict(Counter(binding[n]['demo_id'] for n in names)),seconds=time.monotonic()-begin)
        return [idx[n] for n in names],report
    finally:restore_rng(saved)

def store(c,t,benchmark):
    begin=time.monotonic();root=Path(c.run_root);mfile=root/'storage'/f'task{t}.json'
    paths=mlr._list_pt(str(O/'features'/str(t)));k=math.ceil(.5*len(paths))
    if mfile.exists():
        m=json.loads(mfile.read_text());assert m['contract_sha256']==sha(O/'CONTRACT.json') and m['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
    else:
        before=hashes(Path(c.buffer_dir));assert not any(n.startswith(f't{t}__') for n in before)
        assert len(before)+k<=mlr.BUFFER_CAPACITY
        demos=json.loads((O/'FEATURES.json').read_text())['demos'][str(t)]
        with h5py.File(str(Path(c.folder)/benchmark.get_task_demonstration(t))) as f:population=np.concatenate([f['data'][d]['actions'][:] for d in demos])
        binding=json.loads((O/'bindings'/f'task{t}.json').read_text())['windows']
        selected,report=select(paths,k,population,binding)
        preflight=json.loads((O/'selection_preflight'/f'task{t}.json').read_text())
        assert report['original_ids']==preflight['original_ids'] and report['selected_ids']==preflight['selected_ids']
        assert len(set(selected))==len(selected)==k
        m=dict(task=t,K=k,current_count=len(paths),before=before,selected=[dict(source=p,destination=f't{t}__{Path(p).name}',sha256=sha(p)) for p in selected],selection=report,contract_sha256=sha(O/'CONTRACT.json'),modal_sha256=sha(O/'MODAL_CONTRACT.json'))
        write(mfile,m)
    buffer=Path(c.buffer_dir);buffer.mkdir(parents=True,exist_ok=True)
    allowed=dict(m['before']);allowed.update({r['destination']:r['sha256'] for r in m['selected']})
    actual=hashes(buffer);assert set(actual)<=set(allowed) and all(allowed[n]==h for n,h in actual.items())
    for r in m['selected']:
        src=Path(r['source']);dst=buffer/r['destination'];assert sha(src)==r['sha256']
        if dst.exists():assert sha(dst)==r['sha256']
        else:
            tmp=dst.with_suffix('.copying');shutil.copy2(src,tmp);assert sha(tmp)==r['sha256'];tmp.replace(dst)
    after=hashes(buffer);assert after==allowed
    result=dict(counts={str(i):sum(n.startswith(f't{i}__') for n in after) for i in range(6,t+1)},total=len(after),hashes=after,modal_sha256=sha(O/'MODAL_CONTRACT.json'),seconds=time.monotonic()-begin)
    write(root/f'buffer_after_task{t}.json',result);return result
