"""Real payload admission, with independent all-removal-cost verification."""
import os,json,time,random,pickle
from pathlib import Path
import numpy as np,torch,h5py
from coreset_exp.common import O,sha,write,hashes,rng
from ra_exp.storage import select
from libero.lifelong.selectors.greedy_kcenter import Distance
def independent_verify(ids,report,D):
    lookup={n:i for i,n in enumerate(ids)};chosen=[lookup[n] for n in report['original_ids']];anchors=set(report['anchor_ids'])
    pending=[n for n in report['anchor_ids'] if n not in report['original_ids']]
    assert len(pending)==len(report['swaps'])
    for name,swap in zip(pending,report['swaps']):
        assert name==swap['incoming'];x=lookup[name];aug=chosen+[x]
        block=D[:,aug];owner=block.argmin(axis=1);ordered=np.sort(block,axis=1);first=ordered[:,0];second=ordered[:,1]
        winner=None
        for pos,old in sorted(enumerate(chosen),key=lambda z:ids[z[1]]):
            if ids[old] in anchors:continue
            nearest=np.where(owner==pos,second,first)
            r,m=float(nearest.max()),float(nearest.mean());oldname=ids[old]
            if winner is None or r<winner[0]-1e-12 or (abs(r-winner[0])<=1e-12 and m<winner[1]-1e-12) or (abs(r-winner[0])<=1e-12 and abs(m-winner[1])<=1e-12 and oldname<winner[2]):winner=(r,m,oldname,pos)
        assert winner[2]==swap['outgoing'] and winner[3]==swap['parent_center_slot']
        chosen[winner[3]]=x;v=D[:,chosen].min(axis=1)
        assert abs(float(v.max())-swap['radius_after'])<1e-12 and abs(float(v.mean())-swap['mean_after'])<1e-12
    assert [ids[i] for i in chosen]==report['selected_ids']
    assert len(set(report['selected_ids'])-set(report['original_ids']))==len(pending)
def main():
    assert not (O/'OFFLINE_ACCEPTANCE.json').exists()
    start=time.time();P=Path(os.environ['RA_PARENT']);fm=json.loads((O/'FEATURES.json').read_text());tasks=json.loads((O/'TASKS.json').read_text());results={};before_hash=hashes(O/'features')
    for t in range(6,10):
        paths=[str(p) for p in sorted((O/'features'/str(t)).glob('*.pt'))];ids=[Path(p).name for p in paths];q=(len(paths)+1)//2
        with h5py.File(tasks[str(t)]['paths'][0]) as f:population=np.concatenate([f['data'][d]['actions'][:] for d in fm['demos'][str(t)]])
        binding=json.loads((O/'bindings'/f'task{t}.json').read_text())['windows']
        rb=rng();chosen,report=select(paths,q,population,binding);ra=rng()
        assert rb['python']==ra['python'] and pickle.dumps(rb['numpy'])==pickle.dumps(ra['numpy'])
        assert torch.equal(rb['cpu'],ra['cpu']) and all(torch.equal(a,b) for a,b in zip(rb['cuda'],ra['cuda']))
        old=json.loads((P/'coreset/storage'/f'task{t}.json').read_text());rnd=json.loads((P/'random/storage'/f'task{t}.json').read_text())
        assert report['original_ids']==[Path(x['source']).name for x in old['selected']]
        assert report['random_draw_ids']==[Path(x['source']).name for x in rnd['selected']]==random.Random(42).sample(ids,q)
        data=[torch.load(p,map_location='cpu') for p in paths]
        d=Distance(np.array([x['feature'][0,:,1:3,:].numpy() for x in data]),np.array([x['actions'][0,-1,:].numpy() for x in data]),population,42)
        independent_verify(ids,report,d.distances(np.arange(len(ids))))
        def desc(names):
            rr=[binding[n] for n in names]
            return dict(n=len(rr),positive=sum(r['last_action'][-1]>0 for r in rr),negative=sum(r['last_action'][-1]<0 for r in rr),zero=sum(r['last_action'][-1]==0 for r in rr),unique_times=len({(r['demo_id'],r['target']) for r in rr}),endpoint_windows=sum(r['target']==r['length']-1 for r in rr),demo_support=len({r['demo_id'] for r in rr}))
        report['diagnostics']={n:desc(v) for n,v in [('full',ids),('original',report['original_ids']),('selected',report['selected_ids']),('anchors',report['anchor_ids'])]}
        write(O/'offline_selection'/f'task{t}.json',report)
        results[str(t)]=dict(N=len(ids),q=q,anchors=report['anchor_count'],swaps=report['swap_count'],radius_before=report['max_before'],radius_after=report['max_after'],mean_before=report['mean_before'],mean_after=report['mean_after'],independent_all_removal_choices=True,random_parent_order=True,coreset_parent_order=True)
        print(json.dumps(dict(task=t,**results[str(t)])),flush=True)
    assert hashes(O/'features')==before_hash==fm['hashes']
    assert sum(results[str(t)]['swaps'] for t in (6,7,8))>0
    write(O/'OFFLINE_ACCEPTANCE.json',dict(status='PASS',tasks=results,inputs_unchanged=True,model_forward=0,optimizer_calls=0,environment_steps=0,seconds=time.time()-start))
if __name__=='__main__':main()
