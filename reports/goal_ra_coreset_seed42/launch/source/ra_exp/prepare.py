"""Read-only parent/resource binding; new experiment directories only."""
import json,os,shutil,time
from pathlib import Path
from coreset_exp.common import O,sha,write,hashes
def main():
    P=Path(os.environ['RA_PARENT']);PS=Path(os.environ['RA_PARENT_SOURCE'])
    assert not O.exists(),'Do not repeat preparation of an existing run'
    pc=json.loads((P/'CONTRACT.json').read_text());fm=json.loads((P/'FEATURES.json').read_text())
    assert json.loads((P/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'
    assert sha(P/'MODAL_CONTRACT.json')==pc['modal_sha256']
    assert hashes(P/'features')==fm['hashes'] and fm['contract_sha256']==sha(P/'CONTRACT.json')
    for n,h in {**pc['inputs'],**pc['configs']}.items():assert sha(n)==h,n
    for n,h in pc['source_sha256'].items():assert sha(PS/n)==h,n
    protected={str(p):sha(p) for p in P.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pth') and not {'features','buffer'}&set(p.parts)}
    O.mkdir(parents=True)
    for n in ['MODAL_CONTRACT.json','FEATURES.json','TASKS.json','PRETRAIN_METADATA.json']:shutil.copy2(P/n,O/n)
    (O/'features').symlink_to(P/'features',target_is_directory=True)
    BS=Path(os.environ['SC_BINDING_SOURCE']);bc=json.loads((BS/'CONTRACT.json').read_text());bindings={}
    for t in range(6,10):
        p=BS/'bindings'/f'task{t}.json';h=sha(p);assert bc['files'][str(p)]==h
        obj=json.loads(p.read_text());assert len(obj['windows'])==[997,889,904,1969][t-6]
        for name,r in obj['windows'].items():assert sha(O/'features'/str(t)/name)==r['sha256']
        (O/'bindings').mkdir(exist_ok=True);shutil.copy2(p,O/'bindings'/p.name);bindings[str(p)]=h
    for name,src in [('ra_coreset','coreset'),('eval','eval')]:
        c=json.loads((P/f'resolved_{src}.json').read_text())
        c.update(run_root=str(O/'ra_coreset'),feature_dir=str(O/'features'),buffer_dir=str(O/'ra_coreset/buffer'),experiment_dir=str(O/'ra_coreset/models'),contract=str(O/'CONTRACT.json'),selector='ra_coreset')
        write(O/f'resolved_{name}.json',c)
    write(O/'PARENT_BINDING.json',dict(parent=str(P),source=str(PS),contract_sha256=sha(P/'CONTRACT.json'),feature_manifest_sha256=sha(P/'FEATURES.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),protected={**protected,**bindings},pretrain_use_state='unknown',effective_use_state=True))
    write(O/'RUN_STATUS.json',dict(status='PREPARED_NOT_FROZEN',time=time.time()))
    print('PREPARED: all original input/feature/source and binding hashes verified')
if __name__=='__main__':main()
