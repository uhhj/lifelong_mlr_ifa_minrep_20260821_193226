"""Read-only parent binding and new run preparation. Never exports features."""
import json,os,time,collections,copy,shutil
from pathlib import Path
import numpy as np,torch,h5py
from coreset_exp.common import O,S,SEED,sha,write,source_hashes,hashes
from sc_exp.selection import prepare

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
    binding_source=Path(os.environ['SC_BINDING_SOURCE'])
    offline=Path(os.environ['SCV2_OFFLINE'])
    assert json.loads((offline/'ACCEPTANCE.json').read_text())['status']=='OFFLINE_PASS'
    assert json.loads((offline/'INDEPENDENT_GEOMETRY.json').read_text())['status']=='PASS'
    summaries={}
    for t in range(6,10):
        p=binding_source/'bindings'/f'task{t}.json'
        expected=json.loads((offline/'INPUT_HASHES.json').read_text())[str(p)]
        assert sha(p)==expected
        (O/'bindings').mkdir(exist_ok=True);shutil.copy2(p,O/'bindings'/p.name)
        result=json.loads((offline/f'task{t}.json').read_text())
        assert result['original_ids']==[Path(x['source']).name for x in json.loads((P/'coreset/storage'/f'task{t}.json').read_text())['selected']]
        write(O/'offline_selection'/f'task{t}.json',result)
        summaries[str(t)]={k:result[k] for k in ['N','q','before','after','retained_original']}
    # Keep the feature/modal provenance bytes unchanged; SC has a separate training contract.
    for name,parent_name in [('sc_coreset','coreset'),('eval','eval')]:
        c=json.loads((P/f'resolved_{parent_name}.json').read_text())
        c.update(run_root=str(O/'sc_coreset'),feature_dir=str(O/'features'),buffer_dir=str(O/'sc_coreset/buffer'),experiment_dir=str(O/'sc_coreset/models'),contract=str(O/'CONTRACT.json'),selector='sc_coreset')
        write(O/f'resolved_{name}.json',c)
    write(O/'PARENT_BINDING.json',dict(parent=str(P),source=str(parent_source),contract_sha256=sha(P/'CONTRACT.json'),feature_manifest_sha256=sha(P/'FEATURES.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),protected=protected,supervision=summaries,pretrain_use_state='unknown',effective_use_state=True))
    write(O/'RUN_STATUS.json',dict(status='PREPARED_NOT_FROZEN',time=time.time()))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
