"""External read-only independent acceptance after pipeline exit, no policy load."""
import json, csv, hashlib, os, subprocess
from pathlib import Path
from collections import defaultdict, Counter
O=Path('/data/mlr_ifa/experiments/E1U_uniform_pool_seed42_20261009')
S=Path('/data/mlr_ifa/experiment_workspaces/E1U_uniform_pool_20261009/source')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def metrics(r):
    auc=sum(sum(r[k][j] for k in range(j,4))/(4-j) for j in range(4))/4
    fwt=sum(r[j][j] for j in range(4))/4
    nbt=sum(sum(r[j][j]-r[k][j] for k in range(j+1,4))/(3-j) for j in range(3))/3
    return dict(auc=auc,fwt=fwt,nbt=nbt)
def main():
    assert (O/'pipeline.exit').read_text().strip()=='0'
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    leftovers=[x for x in processes.splitlines() if ('protective_e1.train' in x or 'protective_e1.evaluate' in x or 'protective_e1/pipeline.sh' in x) and 'ps -eo' not in x]
    assert not leftovers,leftovers
    contract=json.loads((O/'CONTRACT.json').read_text());assert contract['name']=='E1U_uniform_pool'
    assert contract['settings']['pool_weighting']=='uniform'
    assert all(sha(S/k)==v for k,v in contract['source_sha256'].items())
    old=json.loads((O/'E1_READONLY_BEFORE.json').read_text());assert all(sha(k)==v for k,v in old.items())
    parity=json.loads((O/'FEATURE_PARITY.json').read_text());assert sum(map(len,parity['features'].values()))==5921
    assert all(sha(O/'features'/t/n)==v for t,files in parity['features'].items() for n,v in files.items())
    train=json.loads((O/'TRAINING_COMPLETE.json').read_text());assert len(train['checkpoints'])==28
    assert all(sha(O/'run_001'/n)==v for n,v in train['checkpoints'].items())
    curve=list(csv.DictReader((O/'run_001/paper_repro_train_curve.csv').open()))
    assert len(curve)==400 and {(int(x['stage']),int(x['epoch'])) for x in curve}=={(s,e) for s in range(6,10) for e in range(100)}
    assert sum(int(r['batches']) for r in curve)==105500
    tables={};counts={}
    for view,expected in [('primary',200),('secondary',680)]:
        records=[json.loads(x) for x in (O/('trials_'+view+'.jsonl')).read_text().splitlines()]
        assert len(records)==expected
        cells=defaultdict(list)
        for r in records:
            assert type(r['success']) is bool;cells[r['stage'],r['epoch'],r['task']].append(r)
        assert all(len(v)==20 and sorted(x['initial_state_index'] for x in v)==list(range(20)) for v in cells.values())
        selected={s:99 if view=='primary' else max([70,75,80,85,90,95,99],key=lambda e:(sum(r['success'] for r in cells[s,e,s]),e)) for s in range(6,10)}
        matrix=[[None]*4 for _ in range(4)]
        for s in range(6,10):
            for t in range(6,s+1):matrix[s-6][t-6]=sum(r['success'] for r in cells[s,selected[s],t])*5.
        published=json.loads((O/('metrics_'+view+'.json')).read_text());assert matrix==published['r']
        tables[view]=dict(matrix=matrix,metrics=metrics(matrix),selected_epochs=selected)
        attempts=[json.loads(x) for x in (O/('attempts_'+view+'.jsonl')).read_text().splitlines()]
        begin=[r for r in attempts if r['event']=='vector_attempt_start'];end=[r for r in attempts if r['event']=='vector_attempt_complete']
        key=lambda r:(r['stage'],r['epoch'],r['task'],r['group'])
        assert len(begin)==len(end)==expected//10 and Counter(map(key,begin))==Counter(map(key,end))
        for r in end:
            original=sorted([a for a in cells[r['stage'],r['epoch'],r['task']] if a['trial']//10==r['group']],key=lambda a:a['trial'])
            assert [a['success'] for a in original]==r['successes']
        counts[view]=dict(formal=expected,attempted_lanes=len(begin)*10)
    pairs=[json.loads(p.read_text()) for p in (O/'pairing').glob('*.json')];assert len(pairs)==400
    audit=json.loads((O/'UNIFORM_AUDIT.json').read_text())
    assert audit['versions']['E1U']['auc']==tables['primary']['metrics']['auc']
    improved=tables['primary']['metrics']['auc']>audit['versions']['E0']['auc']
    result=dict(status='SEED42_COMPLETE_EXTENSION_PENDING' if improved else 'COMPLETED',independent_audit='PASS',
        pipeline_exit=0,residual_processes=leftovers,counts=counts,tables=tables,
        strict_pairing=all(p['strict_pairing'] for p in pairs),original_files_unchanged=len(old),
        source_unchanged=True,features_unchanged=5921,checkpoints_hashed=28,
        extension_seeds=[43,44] if improved else [],limitations=audit['limits'])
    for name in ('INDEPENDENT_ACCEPTANCE.json','RUN_STATUS.json'):(O/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
