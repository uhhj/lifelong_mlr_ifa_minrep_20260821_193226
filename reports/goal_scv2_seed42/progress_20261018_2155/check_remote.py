"""Read-only progress and actual buffer audit; no model or environment calls."""
import json,time,subprocess,hashlib,statistics
from pathlib import Path
from coreset_exp.common import O,verify,sha,hashes
from sc_exp.validation import same_witness
W=Path('/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018')
P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017')
OUT=Path('/tmp/scv2_progress_20261018_2155')
def js(p):return json.loads(p.read_text())
def lines(p):
    raw=p.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
    return [json.loads(x) for x in raw.splitlines()]
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    OUT.mkdir(exist_ok=False);start=time.time();contract=verify()
    for p,h in contract['files'].items():assert sha(p)==h,p
    root=O/'sc_coreset';epochs=js(root/'epochs.json')
    assert len(epochs)==400 and epochs[-1]['steps']==76100 and sum(x['batches'] for x in epochs)==76100
    assert {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    assert js(O/'TASK6_TENSOR_PARITY.json')['equal']
    assert js(root/'INITIAL_STATE.json')==js(P/'coreset/INITIAL_STATE.json')
    results=[];protected={};expected={};ifa=[]
    for t in range(6,10):
        path=root/'storage'/f'task{t}.json';protected[str(path)]=sha(path)
        m=js(path);off=js(O/'offline_selection'/f'task{t}.json');pre=js(O/'selection_preflight'/f'task{t}.json')
        names=[Path(r['source']).name for r in m['selected']]
        assert names==off['selected_ids']==pre['selected_ids']==m['selection']['selected_ids']
        assert m['selection']['original_ids']==off['original_ids']
        for key in ['before','after','swaps']:same_witness(m['selection'][key],off[key])
        assert m['K']==len(names)==len(set(names))==[499,445,452,985][t-6]
        assert m['before']==expected
        for r in m['selected']:
            assert r['destination']==f't{t}__{Path(r["source"]).name}'
            assert sha(r['source'])==sha(root/'buffer'/r['destination'])==r['sha256']
            expected[r['destination']]=r['sha256']
        assert js(root/f'buffer_after_task{t}.json')['hashes']==expected
        results.append(dict(task=t,K=m['K'],swaps=len(off['swaps']),ordered_ids_equal=True,payload_sha_equal=True,
            selected_ids=names,buffer_sha256={r['destination']:r['sha256'] for r in m['selected']},
            before=off['before'],after=off['after'],storage_manifest_sha256=protected[str(path)]))
        current=js(root/f'ifa_task{t}.json');parent=js(P/'coreset'/f'ifa_task{t}.json')
        ifa.append(dict(task=t,current=current,parent=parent,equal=current==parent))
    assert hashes(root/'buffer')==expected
    assert all(sha(p)==h for p,h in protected.items())
    complete={};starts={};attempts=[]
    for p in sorted((root/'evaluation').glob('attempts*.jsonl')):attempts+=lines(p)
    for row in attempts:
        if row['event']=='vector_attempt_start':starts[row['stage'],row['task'],row['group']]=row
        if row['event']=='vector_attempt_complete':complete[row['stage'],row['task'],row['group']]=row
    durations=[dict(stage=k[0],task=k[1],group=k[2],seconds=row['unix']-starts[k]['unix'],steps=row['steps']) for k,row in sorted(complete.items())]
    trials=[]
    for p in sorted((root/'evaluation').glob('trials*.jsonl')):
        for r in lines(p):
            if (r['stage'],r['task'],r['trial']//10) in complete:trials.append(r)
    assert len(trials)==len(complete)*10
    assert len({(r['stage'],r['task'],r['trial']) for r in trials})==len(trials)
    cells=[]
    for s in range(6,10):
        for t in range(6,s+1):
            rr=[r for r in trials if r['stage']==s and r['task']==t]
            if rr:cells.append(dict(stage=s,task=t,n=len(rr),successes=sum(r['success'] for r in rr),complete=len(rr)==20))
    snapshot=json.loads(subprocess.check_output(['/data/mlr_ifa/env/bin/python',str(W/'monitor_snapshot.py')],text=True))
    snapshot['exits']['supervisor']=(O/'supervisor.exit').read_text().strip() if (O/'supervisor.exit').exists() else None
    train=js(root/'TRAINING_COMPLETE.json')
    ckpts={p.name:sha(p) for p in sorted((root/'models').glob('*ep99.pth'))};assert ckpts==train['checkpoints'] and len(ckpts)==4
    result=dict(status='PARTIAL_EVALUATING',observed_unix=time.time(),snapshot=snapshot,
        epochs=400,steps=76100,train_complete=train,checkpoint_sha256=ckpts,
        task6_tensor_equal=True,initial_state_equal=True,storage=results,ifa=ifa,
        frozen_inputs_pass=True,source_files=len(contract['source_sha256']),frozen_files=len(contract['files']),
        contract_sha256=sha(O/'CONTRACT.json'),buffer_files=len(expected),
        committed_records=len(trials),completed_vectors=len(complete),durations=durations,cells=cells,
        technical_exceptions=sum(r['event']=='technical_exception' for r in attempts),
        current_vectors=[dict(stage=k[0],task=k[1],group=k[2],started=v['unix']) for k,v in starts.items() if k not in complete],
        log_mtimes={p.name:p.stat().st_mtime for p in O.glob('*.log')},audit_seconds=time.time()-start,
        no_model_loads=True,no_optimizer_or_environment_calls=True)
    write('CHECK.json',result);write('trials.json',trials);write('attempts.json',attempts);write('epochs.json',epochs)
    manifest={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.glob('*.json')}
    write('MANIFEST.json',manifest)
    print(json.dumps({k:v for k,v in result.items() if k not in ['storage','ifa','snapshot']}),flush=True)
    print(str(OUT),flush=True)
if __name__=='__main__':main()
