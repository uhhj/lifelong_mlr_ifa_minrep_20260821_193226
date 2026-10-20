"""Read-only parent/resource binding; new experiment directories only."""
import json,os,shutil,time,collections
import numpy as np,torch,h5py
from pathlib import Path
from coreset_exp.common import O,SEED,sha,write,hashes
def main():
    P=Path(os.environ['RA_PARENT']);PS=Path(os.environ['RA_PARENT_SOURCE'])
    assert not O.exists(),'Do not repeat preparation of an existing run'
    pc=json.loads((P/'CONTRACT.json').read_text());fm=json.loads((P/'FEATURES.json').read_text())
    assert SEED in (41,42,43) and pc['seed']==SEED
    assert json.loads((P/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'
    assert sha(P/'MODAL_CONTRACT.json')==pc['modal_sha256']
    assert hashes(P/'features')==fm['hashes'] and fm['contract_sha256']==pc.get('feature_origin_contract_sha256',sha(P/'CONTRACT.json'))
    for n,h in {**pc['inputs'],**pc['configs']}.items():assert sha(n)==h,n
    for n,h in pc['source_sha256'].items():assert sha(PS/n)==h,n
    protected={str(p):sha(p) for p in P.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pth') and not {'features','buffer'}&set(p.parts)}
    O.mkdir(parents=True)
    for n in ['MODAL_CONTRACT.json','FEATURES.json','TASKS.json','PRETRAIN_METADATA.json']:shutil.copy2(P/n,O/n)
    (O/'features').symlink_to(P/'features',target_is_directory=True)
    # Seed-specific cached windows must be bound to their own actual payloads.
    tasks=json.loads((P/'TASKS.json').read_text());bindings={};torch.set_num_threads(1)
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
            a=torch.load(p,map_location='cpu')['actions'][0].numpy().astype(np.float32)
            hits=candidates[a.tobytes()];assert hits,(t,p.name)
            identity={(r['demo_id'],r['target'],r['length']) for r in hits}
            unique=len(identity)==1
            demos={r[0] for r in identity};lengths={r[2] for r in identity}
            demo=next(iter(demos)) if len(demos)==1 else 'AMBIGUOUS'
            target=next(iter(identity))[1] if unique else None
            length=next(iter(lengths)) if len(lengths)==1 else None
            ambiguous+=not unique
            windows[p.name]=dict(sample_id=p.name,sha256=sha(p),demo_id=demo,target=target,length=length,last_action=a[-1].tolist(),possible_starts=[r['start'] for r in hits],candidate_identities=[list(x) for x in sorted(identity)],identity_unique=unique,binding='exact complete 8x7 float32 payload; ambiguous target remains unknown and unused by selection')
        assert len(windows)==[1562,1480,1560,1319][t-6]
        write(O/'bindings'/f'task{t}.json',dict(task=t,hdf5=hdf,hdf5_sha256=sha(hdf),windows=windows,ambiguous_target_identity=ambiguous))
    for name,src in [('ra_coreset','coreset'),('eval','eval')]:
        c=json.loads((P/f'resolved_{src}.json').read_text())
        assert c['seed']==SEED and c['gamma']==.3 and c['benchmark_name']=='LIBERO_OBJECT' and c['policy']['use_state'] is True
        c.update(run_root=str(O/'ra_coreset'),feature_dir=str(O/'features'),buffer_dir=str(O/'ra_coreset/buffer'),experiment_dir=str(O/'ra_coreset/models'),contract=str(O/'CONTRACT.json'),selector='ra_coreset')
        write(O/f'resolved_{name}.json',c)
    write(O/'PARENT_BINDING.json',dict(parent=str(P),source=str(PS),contract_sha256=sha(P/'CONTRACT.json'),feature_manifest_sha256=sha(P/'FEATURES.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),protected={**protected,**bindings},pretrain_use_state='unknown',effective_use_state=True))
    write(O/'RUN_STATUS.json',dict(status='PREPARED_NOT_FROZEN',time=time.time()))
    print('PREPARED: all original input/feature/source and binding hashes verified')
if __name__=='__main__':main()
