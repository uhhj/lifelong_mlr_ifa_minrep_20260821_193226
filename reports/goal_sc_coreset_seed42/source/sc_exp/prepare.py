"""Read-only parent binding and new run preparation. Never exports features."""
import json,os,time,collections,copy,shutil
from pathlib import Path
import numpy as np,torch,h5py
from coreset_exp.common import O,S,SEED,sha,write,source_hashes,hashes
from sc_exp.selection import group_id

def main():
    P=Path(os.environ['SC_PARENT']);assert not (O/'CONTRACT.json').exists();O.mkdir(exist_ok=True,parents=True)
    pc=json.loads((P/'CONTRACT.json').read_text());fm=json.loads((P/'FEATURES.json').read_text());tasks=json.loads((P/'TASKS.json').read_text())
    assert json.loads((P/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'
    assert sha(P/'MODAL_CONTRACT.json')==pc['modal_sha256']
    assert hashes(P/'features')==fm['hashes'] and fm['contract_sha256']==sha(P/'CONTRACT.json')
    for name,h in pc['inputs'].items():assert sha(name)==h,name
    for name,h in pc['configs'].items():assert sha(name)==h,name
    parent_source=Path(os.environ['SC_PARENT_SOURCE'])
    for name,h in pc['source_sha256'].items():assert sha(parent_source/name)==h,name
    protected={str(p):sha(p) for p in P.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pth') and 'features' not in p.parts and 'buffer' not in p.parts}
    for name in ['MODAL_CONTRACT.json','FEATURES.json','TASKS.json','PRETRAIN_METADATA.json']:
        shutil.copy2(P/name,O/name)
    (O/'features').symlink_to(P/'features',target_is_directory=True)
    summaries={}
    torch.set_num_threads(1)
    for t in range(6,10):
        hdf=tasks[str(t)]['paths'][0];candidates=collections.defaultdict(list)
        with h5py.File(hdf) as h:
            for demo in fm['demos'][str(t)]:
                a=h['data'][demo]['actions'][:].astype(np.float32)
                for start in range(len(a)):
                    ix=np.minimum(np.arange(start,start+8),len(a)-1)
                    candidates[a[ix].tobytes()].append(dict(demo_id=demo,start=start,target=int(ix[-1]),length=len(a)))
        windows={};ambiguous=0
        for p in sorted((P/'features'/str(t)).glob('*.pt')):
            a=torch.load(p,map_location='cpu')['actions'][0].numpy().astype(np.float32);hits=candidates[a.tobytes()];assert hits,(t,p.name)
            identity={(r['demo_id'],r['target'],r['length']) for r in hits};assert len(identity)==1,('Ambiguous supervision identity',t,p.name,hits)
            demo,target,length=next(iter(identity));ambiguous+=len(hits)>1
            windows[p.name]=dict(sample_id=p.name,sha256=sha(p),demo_id=demo,target=target,length=length,last_action=a[-1].tolist(),group=group_id(target,length,float(a[-1,-1])),possible_starts=[r['start'] for r in hits],binding='exact complete 8x7 float32 action window; unique demo/target/length')
        assert len(windows)==[997,889,904,1969][t-6]
        write(O/'bindings'/f'task{t}.json',dict(task=t,hdf5=hdf,hdf5_sha256=sha(hdf),windows=windows,ambiguous_start_but_unique_target=ambiguous))
        summaries[str(t)]=dict(n=len(windows),groups=dict(collections.Counter(r['group'] for r in windows.values())),ambiguous_start_but_unique_target=ambiguous)
    # Keep the feature/modal provenance bytes unchanged; SC has a separate training contract.
    for name,parent_name in [('sc_coreset','coreset'),('eval','eval')]:
        c=json.loads((P/f'resolved_{parent_name}.json').read_text())
        c.update(run_root=str(O/'sc_coreset'),feature_dir=str(O/'features'),buffer_dir=str(O/'sc_coreset/buffer'),experiment_dir=str(O/'sc_coreset/models'),contract=str(O/'CONTRACT.json'),selector='sc_coreset')
        write(O/f'resolved_{name}.json',c)
    write(O/'PARENT_BINDING.json',dict(parent=str(P),source=str(parent_source),contract_sha256=sha(P/'CONTRACT.json'),feature_manifest_sha256=sha(P/'FEATURES.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),protected=protected,supervision=summaries,pretrain_use_state='unknown',effective_use_state=True))
    write(O/'RUN_STATUS.json',dict(status='PREPARED_NOT_FROZEN',time=time.time()))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
